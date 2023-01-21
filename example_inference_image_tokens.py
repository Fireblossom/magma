from magma import Magma
import torch
import json
from PIL import Image
from magma.utils import to_cuda_half

model = Magma.from_checkpoint(
    config_path = "configs/MAGMA_v1_facebook_resnet_image_tokens.yml",
    checkpoint_path = "checkpoints/multimodal_facebook_resnet_image_tokens/global_step12500/mp_rank_00_model_states.pt",
    device = 'cuda:0'
)
model.device = torch.device('cuda:0')
img_path = 'example/SPM.06-01-1.png'

ocr = json.load(open('example/SPM.06-01-1.json'))
img = Image.open(img_path)
if img.mode == "RGBA":
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background

img_tensor = model.transforms(img, ocr['img_text'], boxes=ocr['bbox'], return_tensors="pt", padding="max_length", truncation=True, max_length=model.transforms.tokenizer.model_max_length)
img_tensor = to_cuda_half(img_tensor)

import json
metadatas = {}
with open('/home/duan/paddle/arxiv-metadata-oai-snapshot.json') as file:
    for line in file:
        d = json.loads(line)
        if d['title'] in metadatas:
            pass
        else:
            metadatas[d['title']] = d['id']
import re
pattern = r"(\[START_REF\]|\[END_REF\])"
while True:
    captions = model(
        img_tensor, captions=None, inference=True, ref=True,
    )  # [caption1, caption2, ... b]
    ## returns a list of length embeddings.shape[0] (batch size)

    ##  A cabin on a lake
    caption_split = re.split(pattern, captions[0])
    reference = caption_split[-3]
    if reference.strip() in metadatas:
        print(captions[0])