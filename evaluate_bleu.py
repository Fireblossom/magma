import evaluate
import torch
import re
import glob

predictions_caption = []
references_caption = []

predictions_title = []
references_title = []

pattern = r"(\[START_REF\]|\[END_REF\])"
exp = "multimodal_facebook_resnet"
base_path = "checkpoints/"+exp+"/"
pred_record = set()
for file in glob.glob(base_path+'test_output.pt*'):
    print(file)
    results = torch.load(file, map_location='cpu')
    for captions, gold in results:
        for idx in range(len(captions)):
            caption = captions[idx]
            caption_split = re.split(pattern, caption)
            if len(re.findall(pattern, caption)) == 2:
                #normal case
                caption_gen = caption_split[0]
                reference = caption_split[2]
                class_prediction = caption_split[-1]
            elif len(re.findall(pattern, caption)) == 1:
                #caotion too long
                caption_gen = caption_split[0]
                reference = caption_split[2]
                class_prediction = None
            elif len(re.findall(pattern, caption)) == 0:
                #caotion toooo long
                caption_gen = caption_split[0]
                reference = None
                class_prediction = None
            else: # > 2
                caption_gen = caption_split[0]
                reference = caption_split[-3]
                class_prediction = caption_split[-1]
            if not id(gold["captions"][idx][0]) in pred_record:
                predictions_caption.append(caption_gen)
                references_caption.append(gold["captions"][idx])

                predictions_title.append(reference)
                references_title.append(gold["titles"][idx])

                pred_record.add(id(gold["captions"][idx][0]))
            else:
                pass

bleu = evaluate.load("bleu")

import json

with open("result/captions_"+exp+".json", 'w') as file:
    json.dump([[predictions_caption[i], references_caption[i]] for i in range(len(predictions_caption))], file, indent=2)
with open("result/titles_"+exp+".json", "w") as file:
    json.dump([[predictions_title[i], references_title[i]] for i in range(len(predictions_caption))], file, indent=2)

results = bleu.compute(predictions=predictions_caption, references=references_caption)
print(results)
results_title = bleu.compute(predictions=predictions_title, references=references_title)
print(results_title)