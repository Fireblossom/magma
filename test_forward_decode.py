from magma import Magma
import torch

model = Magma('configs/MAGMA_v1_layoutlmv3.yml', device=torch.device("cuda:0"))
model = model.half().cuda()
from train import get_pretraining_datasets
tokenizer, config, transforms = model.tokenizer, model.config, model.transforms
train_dataset, eval_dataset = get_pretraining_datasets(
        config, tokenizer, transforms
    )

from magma.datasets import collate_fn
from torch.utils.data import DataLoader
from magma.utils import cycle
from functools import partial
from torchvision.utils import make_grid
from transformers.tokenization_utils_base import BatchEncoding

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=partial(collate_fn, seq_len=model.seq_len))
train_loader = cycle(train_loader)
images, captions = next(train_loader)

def to_cuda_half(*args):
    cuda_half_args = []
    for x in args:
        if isinstance(x, BatchEncoding):
            for k in x.keys():
                if x[k].dtype in [torch.float32, torch.float16]:
                    x[k] = x[k].half().cuda()
                elif x[k].dtype == torch.long:
                    x[k] = x[k].cuda()
            cuda_half_args.append(x)
        elif isinstance(x, list):
            x_cuda_half = to_cuda_half(*x)
            cuda_half_args.append(x_cuda_half)
        elif isinstance(x, tuple):
            x_cuda_half = to_cuda_half(*x)
            cuda_half_args.append(x_cuda_half)
        else:
            if x.dtype in [torch.float32, torch.float16]:
                cuda_half_args.append(x.half().cuda())
            elif x.dtype == torch.long:
                cuda_half_args.append(x.cuda())
    if len(cuda_half_args) == 1:
        return cuda_half_args[0]
    else:
        return cuda_half_args

images, captions = to_cuda_half(images, captions)
outputs = model.forward(images, captions)

print(outputs)


batch_size = len(images['input_ids'])
input_embeddings = model.image_prefix(images)
asks = [model.tokenizer.encode('Describe the image:')] * batch_size
word_embeddings = model.word_embedding(torch.LongTensor(asks).to(model.device))
input_embeddings = torch.cat(
    (
        input_embeddings,
        word_embeddings[:, : -input_embeddings.shape[1], :],
    ),  # remove padding in the word embedding before concatenating
    dim=1,
)

captions = model(
    images, captions=None, inference=True, ref=True
)  # [caption1, caption2, ... b]
width = min(2, images['pixel_values'].shape[0])
image_grid = make_grid(images['pixel_values'][:width])
caption = ""
for i in range(width):
    caption += f"Caption {i}: \n{captions[i]}\n"
print(image_grid, caption)