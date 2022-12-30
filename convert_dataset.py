# %%
import json
import glob
import os
from tqdm import tqdm
from pathlib import Path

# %%
base_path = '/home/duan/magma_2/dataset/scicap_data/SciCap-Caption-All'
jsons = glob.glob(os.path.join(base_path, '**/*.json'), recursive=True)

# %%
ds_train = []
ds_test = []
ds_val = []

resize_w = 550
resize_h = 800

for file in tqdm(jsons):
    if 'test' in file:
        sub_path = 'test'
    elif 'train' in file:
        sub_path = 'train'
    elif 'val' in file:
        sub_path = 'val'
    else:
        raise 'path error..'

    with open(file) as json_file:
        d = json.load(json_file)
    assert 'paper_title' in d

    if d["contains-subfigure"] is True:
        img_path = os.path.join('/home/duan/magma_2/dataset/scicap_data/', 'SciCap-Yes-Subfig-Img', sub_path, d['figure-ID'])
    else:
        img_path = os.path.join('/home/duan/magma_2/dataset/scicap_data/', 'SciCap-No-Subfig-Img', sub_path, d['figure-ID'])
    
    w, h = d["img_size"]
    resize_factor =  min(resize_w/w , resize_h/h)
    assert h*resize_factor < 1000 and w*resize_factor < 1000, 'resize error'
    corners = [i[0] for i in d["img_text"]]
    bbox = []
    for corner in corners:
        x0 = int(min([i[0] for i in corner])*resize_factor + 100)
        y0 = int(min([i[1] for i in corner])*resize_factor + 100)
        x1 = int(max([i[0] for i in corner])*resize_factor + 100)
        y1 = int(max([i[1] for i in corner])*resize_factor + 100)
        assert max([x0, y0, x1, y1]) < 1000, 'bbox range error'
        bbox.append([x0, y0, x1, y1])
    item = tuple([
        Path(img_path), 
        {
            "captions": d["2-normalized"]["2-1-basic-num"]["sentence"],
            "metadata":{
                "paper_title": d["paper_title"],
                "img_text": [i[1] for i in d["img_text"]],
                "bbox": bbox,
                "img_size": d["img_size"]
            }}
        ]
    )

    if 'test' in file:
        ds_test.append(item)
    elif 'train' in file:
        ds_train.append(item)
    elif 'val' in file:
        ds_val.append(item)
    


# %%
from magma.datasets.convert_datasets import convert_dataset
convert_dataset(data_dir="dataset/scicap/train", ds_iterator=ds_train, mode="cp")
convert_dataset(data_dir="dataset/scicap/test", ds_iterator=ds_test, mode="cp")
convert_dataset(data_dir="dataset/scicap/val", ds_iterator=ds_val, mode="cp")
# %%



