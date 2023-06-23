import json
from tqdm import tqdm
import requests
from requests_futures.sessions import FuturesSession
from concurrent.futures import ThreadPoolExecutor, as_completed
session = FuturesSession(executor=ThreadPoolExecutor(max_workers=5))
import datetime
import time

from seleniumrequests import Chrome 
from selenium.webdriver.chrome.options import Options
import json
import os

exp_file = 'result/merged_multimodal_facebook_resnet_image_tokens.json'
#exp_file = 'result/merged_multimodal_facebook_resnet.json'
#exp_file = 'result/merged_multimodal_transformer_layoulmv3.json'
exp_file = 'result/test.json'
print(exp_file)

chrome_options = Options()
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
#chrome_options.add_argument("--no-sandbox") # linux only
chrome_options.add_argument("--headless")
# chrome_options.headless = True # also works
driver = Chrome(options=chrome_options)
start_url = "https://www.semanticscholar.org/search?q=Multi-channel%20FuD-19%20Lesion%20Segmentation%20on%20CT&sort=relevance"
driver.get(start_url)
driver.implicitly_wait(2)

s2_api_key = 'JfneD5mmBt3tmVpDpltEp3rm9eVDuJcv3fMLtuC8'
header = {'x-api-key': s2_api_key}

def get_recommends(paper_id):
    url = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{}".format(paper_id)
    f1 = session.get(url, headers=header, params={
        #"from": "all-cs",
        "fields": ['title']
    })
    f2 = session.get(url, headers=header, params={
        "from": "all-cs",
        "fields": ['title']
    })
    time.sleep(0.1)
    try:
        #f1.raise_for_status()
        return {
            "new_paper": f1,#.json()["recommendedPapers"],
            "all_paper": f2#.json()["recommendedPapers"]
        }
    except:
        return None

def search_paper(query):
    if query == '':
        query = ' '
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    f = session.get(url, headers=header, params={
        "query": query,
        "fields": ['title']
    })
    time.sleep(0.05)
    #r.raise_for_status()
    return f#.json()

def search_paper_online(query):
    if query == '':
        return None
    body = json.loads('{"queryString":"","page":1,"pageSize":10,"sort":"relevance","authors":[],"coAuthors":[],"venues":[],"yearFilter":null,"requireViewablePdf":false,"fieldsOfStudy":[],"useFallbackRankerService":false,"useFallbackSearchCluster":false,"hydrateWithDdb":true,"includeTldrs":true,"performTitleMatch":true,"includeBadges":true,"getQuerySuggestions":false,"cues":["CitedByLibraryPaperCue"]}')
    body['queryString'] = query
    body = json.dumps(body)
    url = "https://www.semanticscholar.org/api/1/search"
    while True:
        r = driver.request('POST', url, data=body, headers={
            "x-forwarded-for": "4.2.2.2",
            "x-s2-client": "webapp-browser",
            "x-s2-ui-version": "f95dd6b69b7854d48f926f16cfbc194d02ce356d",
            "x-api-key": "JfneD5mmBt3tmVpDpltEp3rm9eVDuJcv3fMLtuC8",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache,no-store,must-revalidate,max-age=-1",
            "content-length": str(len(query)),
            "content-type": "application/json",
            "origin": "https://www.semanticscholar.org",
            "pragma": "no-cache",
            "referer": "https://www.semanticscholar.org/search?q=Multi-channel%20Fusion%20for%20COVID-19%20Lesion%20Segmentation%20on%20CT&sort=relevance",
            "sec-ch-ua": '"Microsoft Edge";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"'
        })
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 202:
            print(r.status_code)
            #return None
            print("reach rate limit, sleep 30s.")
            time.sleep(30)
        else:
            print("find error.")
            return None

def get_future(futures, length):
    complete = [None] * length
    for future in as_completed(futures):
        i = futures[future]
        try:
            data = future.result()
            complete[i] = data.json()
            if 'recommendedPapers' in complete[i]:
                complete[i] = complete[i]['recommendedPapers']
        except:
            pass
    return complete
#requests.post('https://www.semanticscholar.org/api/1/search/', params=json.loads('{"queryString":"Multi-channel Fusion for COVID Lesion Segmentation on CT","page":1,"pageSize":10,"sort":"relevance","authors":[],"coAuthors":[],"venues":[],"yearFilter":null,"requireViewablePdf":false,"fieldsOfStudy":[],"useFallbackRankerService":false,"useFallbackSearchCluster":false,"hydrateWithDdb":true,"includeTldrs":true,"performTitleMatch":true,"includeBadges":true,"getQuerySuggestions":false,"cues":["CitedByLibraryPaperCue"]}')).text
#merged = json.load(open("result/full_result.json", "r"))
if not os.path.isfile(exp_file+'.1'):

    with open(exp_file) as f:
        merged = json.load(f)

    queue_gold = {}
    queue_predict = {}
    for i, item in enumerate(tqdm(merged)):
        titles = item["title"]
        if 's2' not in item:
            item['s2'] = {
                'gold':{"s2item": None, "recommend": None},
                'predict':{"s2item": None, "recommend": None}
            }

        if not 's2' in item or item['s2']['gold']["s2item"] is None:
            result_gold = search_paper(titles['gold'])
            assert result_gold not in queue_gold
            queue_gold[result_gold] = i
        if not 's2' in item or item['s2']['predict']["s2item"] is None:
            result_predict = search_paper(titles['predict'])
            assert result_predict not in queue_predict
            queue_predict[result_predict] = i

    complete_gold = get_future(queue_gold, len(merged))
    complete_predict = get_future(queue_predict, len(merged))

    full_result1 = []
    for i, item in enumerate(tqdm(merged)):
        titles = item["title"]
        if 's2' not in item:
            item['s2'] = {
                'gold':{"s2item": None, "recommend": None},
                'predict':{"s2item": None, "recommend": None}
            }

        if not 's2' in item or item['s2']['gold']["s2item"] is None:
            result_gold = complete_gold[i]
            if (not result_gold is None) and result_gold['total'] > 0:
                r_gold = result_gold['data'][0]
                #recommends_gold = get_recommends(r_gold["paperId"])
            else:
                result_gold = search_paper_online(titles['gold'])
                if not result_gold is None and result_gold['totalResults'] > 0:
                    r_gold = result_gold['results'][0]
                    r_gold = {"paperId": r_gold['id'], "title": r_gold['title']['text']}
                    #recommends_gold = get_recommends(r_gold["paperId"])
                else:
                    r_gold = None
                    #recommends_gold = None
            item['s2']['gold'] = {
                "s2item": r_gold,
                "recommend": None,
                "date": str(datetime.datetime.now())
            }
            
        if not 's2' in item or item['s2']['predict']["recommend"] is None:
            result_predict = complete_predict[i]
            if (not result_predict is None) and result_predict['total'] > 0:
                r_predict = result_predict['data'][0]
                #recommends_predict = get_recommends(r_predict["paperId"])
            else:
                result_predict = search_paper_online(titles['predict'])
                if not result_predict is None and result_predict['totalResults'] > 0:
                    r_predict = result_predict['results'][0]
                    r_predict = {"paperId": r_predict['id'], "title": r_predict['title']['text']}
                    #recommends_predict = get_recommends(r_predict["paperId"])
                else:
                    r_predict = None
                    #recommends_predict = None
            item['s2']['predict'] = {
                "s2item": r_predict,
                "recommend": None,
                "date": str(datetime.datetime.now())
            }
        full_result1.append(item)
    driver.quit()
    with open(exp_file + ".1", "w") as file:
        json.dump(full_result1, file, indent=2)
else:
    with open(exp_file + ".1", "r") as file:
        full_result1 = json.load(file)

full_result2 = []
queue_gold = {}
queue_predict = {}

queue_gold_new = {}
queue_predict_new = {}
for i, item in enumerate(tqdm(full_result1)):
    if item['s2']['gold']["recommend"] is None:
        paper_id_gold = item['s2']['gold']["s2item"]["paperId"]
        recommend_gold = get_recommends(paper_id_gold)
        queue_gold_new[recommend_gold["new_paper"]] = i
        queue_gold[recommend_gold["all_paper"]] = i

    if item['s2']['predict']["recommend"] is None:
        paper_id_predict = item['s2']['predict']["s2item"]["paperId"]
        recommend_predict = get_recommends(paper_id_predict)
        queue_predict_new[recommend_predict["new_paper"]] = i
        queue_predict[recommend_predict["all_paper"]] = i

complete_gold = get_future(queue_gold, len(full_result1))
complete_predict = get_future(queue_predict, len(full_result1))
complete_gold_new = get_future(queue_gold_new, len(full_result1))
complete_predict_new = get_future(queue_predict_new, len(full_result1))

full_result = []
for i, item in enumerate(tqdm(full_result1)):
    if item['s2']['gold']["recommend"] is None:
        recommends_gold = {
            "new_paper": complete_gold_new[i],
            "all_paper": complete_gold[i]
        }
        item['s2']['gold']["recommend"] = recommends_gold

    if item['s2']['predict']["recommend"] is None:
        recommends_predict = {
            "new_paper": complete_predict_new[i],
            "all_paper": complete_predict[i]
        }
        item['s2']['predict']["recommend"] = recommends_predict
    full_result.append(item)

with open(exp_file+'.final', 'w') as f:
    json.dump(full_result, f, indent=2)