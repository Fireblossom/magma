import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import Image as img
from PIL.Image import DecompressionBombError
from PIL import UnidentifiedImageError
import json
from pathlib import Path
from transformers.tokenization_utils_base import BatchEncoding

from tqdm import tqdm
from typing import List, Tuple, Generator
import random
from multiprocessing import Pool, cpu_count

from PIL import Image
from typing import Tuple
from torchtyping import TensorType
import traceback


def read_jsonl(filename: str) -> Generator[List, None, None]:
    """
    Iterator over data from a jsonl file
    """
    with open(filename) as file:
        for line in file:
            yield json.loads(line.rstrip("\n|\r"))


def read_img_captions(filename: str) -> List[Tuple[str, str]]:
    """
    Yields image_path, image_caption from cc jsonl files
    """
    img_captions = []
    for item in read_jsonl(filename):
        if not "N/A" in item[-2:]:
            img_captions.append((item[-1], item[-2]))
    return img_captions


def load_json(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except Exception:
        print(f"ERROR: Error loading json file {filename}")
        traceback.print_exc()


def _read_image_data(data_dir):
    image_data = []
    img_data_dir = data_dir / "image_data"
    paths = _load_paths(data_dir)
    pbar = tqdm(
        paths,
        desc=f"loading dataset from {str(data_dir)}",
    )
    # read data with multiprocessing
    with Pool(min(16,cpu_count())) as pool:
        for img_data in pool.imap(load_json, pbar):
            if img_data is not None:
                image_data.append(img_data)
    return image_data


def _load_paths(data_dir, sort=True):
    paths = []
    img_data_dir = data_dir / "image_data"
    for p in tqdm(
        Path(img_data_dir).glob("*/*.json"),
        desc=f"loading dataset paths from {str(data_dir)}",
    ):
        paths.append(p)
    return sorted(paths)


class LazyLoader:
    def __init__(self, data_dir):
        self.paths = _load_paths(data_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = load_json(self.paths[idx])
        if data is None:
            return self[random.randint(0, len(self) - 1)]
        return data


class ImgCptDataset(Dataset):
    """
    Dataset which loads image caption data from our standard format and transforms them into tensors that can be input to the model.
    Images are expected to be stored in data_dir/images, image data in data_dir/image_data and each data item is a json file with format {"image_path": img_path, "captions": [caption1, caption2,...], "metadata":{...}}
    """

    def __init__(
        self, data_dir, tokenizer, transforms, seq_len=2048, load_data_in_memory=False, config=None
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.seq_len = seq_len
        self.load_data_in_memory = load_data_in_memory
        if self.load_data_in_memory:
            self.data = _read_image_data(self.data_dir)
        else:
            self.data = LazyLoader(self.data_dir)
        self.config = config
        if not self.config.cache_prefix is None:
            self.cache_path = Path('./cache') / self.config.cache_prefix
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.cached = [None]*len(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx
    ) -> Tuple[TensorType["b", "c", "h", "w"], TensorType["b", "s"]]:
        img_data = self.data[idx]
        try:
            try:
                img_path = self.data_dir / img_data["image_path"]
            except KeyError as e:
                # if no image path is found, assume path is same as .json, but .jpg
                if not self.load_data_in_memory:
                    p = self.data.paths[idx]
                    img_path = (
                        self.data_dir
                        / "images"
                        / Path(p.parent).name
                        / Path(p.name).with_suffix(".jpg")
                    )
                else:
                    raise e
            if not self.config.cache_prefix is None:
                cache_file = self.cache_path / (img_path.name+'.pt')
                if not self.cached[idx] is None:
                    img_tensor, caption_tensor = self.cached[idx]
                    return img_tensor, caption_tensor
                elif cache_file.is_file():
                    img_tensor, caption_tensor = torch.load(cache_file)
                    return img_tensor, caption_tensor

            img = Image.open(img_path)
            if ("layoutlmv3" in self.config.encoder_name) or self.config.image_token_embedding:
                # transforms is a layoutlmv3 processor
                words = img_data["metadata"]["img_text"]
                bboxes = img_data["metadata"]["bbox"]
                img_tensor = self.transforms(img, words, boxes=bboxes, return_tensors="pt", padding="max_length", truncation=True, max_length=self.transforms.tokenizer.model_max_length)
            else:
                img_tensor = self.transforms(img)

            if not "facebook" in self.config.lm_name:
                caption = random.choice(img_data["captions"])
            else:
                caption = random.choice(img_data["captions"]) + ' [START_REF] ' + img_data["metadata"]["paper_title"] + ' [END_REF]'

            caption_tensor = self.tokenizer.encode(
                caption,
                return_tensors="pt",
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
            )

            if not self.config.cache_prefix is None:
                if not cache_file.is_file():
                    torch.save([img_tensor, caption_tensor], cache_file)
                    self.cached[idx] = [img_tensor, caption_tensor]
            return img_tensor, caption_tensor
        except (
            UnidentifiedImageError,
            OSError,
            DecompressionBombError,
            IndexError,
        ) as e:
            # return random index if image is corrupt
            print(f"Warning: Could not load image {str(img_path)}")
            return self[random.randint(0, len(self) - 1)]


def collate_fn(batch_data: List[Tuple[torch.Tensor, torch.Tensor]], seq_len=2048):
    if isinstance(batch_data[0][0], BatchEncoding):
        batch_captions = [i[1] for i in batch_data]
        batch_images = [i[0] for i in batch_data]
        batch_encodings = batch_images[0]
        for image_encodeing in batch_images[1:]:
            for k in batch_encodings.keys():
                #print(type(batch_encodings), type(image_encodeing), k)
                batch_encodings[k] = torch.cat((batch_encodings[k], image_encodeing[k]), dim=0) 
        return batch_encodings, torch.cat([i[:, :seq_len] for i in batch_captions])
    else:
        all_images, all_captions = list(
            zip(*batch_data)
        )  # [(img1, caption1), (img2, caption2), ... ] -> [(img1, img2, ... ), (caption1, caption2, ... )]
        return torch.cat(all_images), torch.cat([i[:, :seq_len] for i in all_captions])
