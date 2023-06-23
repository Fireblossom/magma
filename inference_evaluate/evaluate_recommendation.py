import json
import os
from tqdm import tqdm

if not os.path.isfile("arxiv_dump.json"):
    titles = {}
    with open("arxiv-metadata-oai-snapshot.json") as file:
        for line in file:
            d = json.loads(line)
            title = d['title']
            titles[title] = d['id']
    with open("arxiv_dump.json", "w") as file:
        json.dump(titles, file,indent=2)
else:
    titles = json.load(open("arxiv_dump.json"))
print("metadata loaded")
merged = json.load(open("result/merged_multimodal_transformer_layoulmv3.json.final", "r"))

count = {
    "gold": 0,
    "predict": 0,
    "none": 0,
    "both": 0
}
hit = {
    "new": 0,
    "all": 0
}
overlaps_new = [0] * len(merged)
count_new = 0
overlaps_all = [0] * len(merged)
count_all = 0
arxiv_hit = 0

for i, paper in enumerate(tqdm(merged)):
    if paper['title']['predict'] and paper['title']['predict'].strip() in titles:
        arxiv_hit += 1

    gold =  paper['s2']['gold']["s2item"] is None
    predict = paper['s2']['predict']["s2item"] is None

    if gold and predict:
        count['none'] += 1
    elif gold:
        count['gold'] += 1
    elif predict:
        count['predict'] += 1
    else:
        count['both'] += 1
        if paper['s2']['gold']['recommend']['new_paper']:
            recommend_gold = [p['paperId'] for p in paper['s2']['gold']['recommend']['new_paper']['recommendedPapers']]
            recommend_gold_set = set(recommend_gold)
            if paper['s2']['predict']['s2item']['paperId'] in recommend_gold_set:
                hit['new'] += 1
            if paper['s2']['predict']['recommend']['new_paper']:
                recommend_predict = [p['paperId'] for p in paper['s2']['predict']['recommend']['new_paper']['recommendedPapers']]
                recommend_predict_set = set(recommend_predict)
                count_new += 1
                overlaps_new[i] = len(recommend_gold_set.intersection(recommend_predict_set))

        if paper['s2']['gold']['recommend']['all_paper']:
            recommend_gold = [p['paperId'] for p in paper['s2']['gold']['recommend']['all_paper']['recommendedPapers']]
            recommend_gold_set = set(recommend_gold)
            if paper['s2']['predict']['s2item']['paperId'] in recommend_gold_set:
                hit['all'] += 1
            if paper['s2']['predict']['recommend']['all_paper']:
                recommend_predict = [p['paperId'] for p in paper['s2']['predict']['recommend']['all_paper']['recommendedPapers']]
                recommend_predict_set = set(recommend_predict)
                count_all += 1
                overlaps_all[i] = len(recommend_gold_set.intersection(recommend_predict_set))

print(arxiv_hit, '/', len(merged))
print(count)
print(hit['new']/count_new, hit['all']/count_all)
print(sum(overlaps_new)/count_new)
print(sum(overlaps_all)/count_all)