import json
from tqdm import tqdm
import requests
import datetime
import time
import os
from seleniumrequests import Chrome 
from selenium.webdriver.chrome.options import Options
import json

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
    r1 = requests.get(url, headers=header, params={
        #"from": "all-cs",
        "fields": ['title']
    })
    r2 = requests.get(url, headers=header, params={
        "from": "all-cs",
        "fields": ['title']
    })
    ret = {
        "new_paper": None,
        "all_paper": None
    }
    try:
        r1.raise_for_status()
        ret['new_paper'] = r1.json()
    except:
        pass
    try:
        r2.raise_for_status()
        ret['all_paper'] = r2.json()
    except:
        pass
    return ret

def search_paper(query):
    if query == '':
        raise
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    r = requests.get(url, headers=header, params={
        "query": query,
        "fields": ['title']
    })
    try:
        r.raise_for_status()
        return r.json()
    except:
        return {'total': 0}
    

def search_paper_online(query):
    if query == '' or query is None:
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
    
#requests.post('https://www.semanticscholar.org/api/1/search/', params=json.loads('{"queryString":"Multi-channel Fusion for COVID Lesion Segmentation on CT","page":1,"pageSize":10,"sort":"relevance","authors":[],"coAuthors":[],"venues":[],"yearFilter":null,"requireViewablePdf":false,"fieldsOfStudy":[],"useFallbackRankerService":false,"useFallbackSearchCluster":false,"hydrateWithDdb":true,"includeTldrs":true,"performTitleMatch":true,"includeBadges":true,"getQuerySuggestions":false,"cues":["CitedByLibraryPaperCue"]}')).text
#merged = json.load(open("result/full_result.json", "r"))
import sys
assert len(sys.argv) == 2
exp_file = sys.argv[1]
#exp_file = "result/merged_multimodal_facebook_resnet.json"
#exp_file = "result/merged_multimodal_transformer_layoulmv3.json"
print(exp_file)
if os.path.isfile(exp_file+'.break'):
    with open(exp_file+'.break') as f:
        merged = json.load(f)
else:
    with open(exp_file) as f:
        merged = json.load(f)

full_result = []
try:
    for item in tqdm(merged):
        titles = item["title"]

        if not 's2' in item or item['s2']['gold']["recommend"] is None:
            result_gold = search_paper(titles['gold'])
            if 'data' in result_gold and result_gold['total'] > 0:
                r_gold = result_gold['data'][0]
                recommends_gold = get_recommends(r_gold["paperId"])
            else:
                result_gold = search_paper_online(titles['gold'])
                if not result_gold is None and result_gold['totalResults'] > 0:
                    r_gold = result_gold['results'][0]
                    r_gold = {"paperId": r_gold['id'], "title": r_gold['title']['text']}
                    recommends_gold = get_recommends(r_gold["paperId"])
                else:
                    r_gold = None
                    recommends_gold = None
            if 's2' not in item:
                item['s2'] = {
                    'gold':{"recommend":None},
                    'predict':{"recommend":None}
                }
            item['s2']['gold'] = {
                "s2item": r_gold,
                "recommend": recommends_gold,
                "date": str(datetime.datetime.now())
            }


        if not 's2' in item or item['s2']['predict']["recommend"] is None:
            result_predict = search_paper(titles['predict'])
            if 'data' in result_predict and result_predict['total'] > 0:
                r_predict = result_predict['data'][0]
                recommends_predict = get_recommends(r_predict["paperId"])
            else:
                result_predict = search_paper_online(titles['predict'])
                if not result_predict is None and result_predict['totalResults'] > 0:
                    r_predict = result_predict['results'][0]
                    r_predict = {"paperId": r_predict['id'], "title": r_predict['title']['text']}
                    recommends_predict = get_recommends(r_predict["paperId"])
                else:
                    r_predict = None
                    recommends_predict = None
            item['s2']['predict'] = {
                "s2item": r_predict,
                "recommend": recommends_predict
            }
            
            

        """item['s2'] = {
            "gold": {
                "s2item": r_gold,
                "recommend": recommends_gold
            },
            "predict": {
                "s2item": r_predict,
                "recommend": recommends_predict
            },
            "date": str(datetime.datetime.now())
        }"""
        print(item)
        full_result.append(item)

    driver.quit()
    with open(exp_file+'.final', 'w') as f:
        json.dump(full_result, f, indent=2)
except Exception as e:
    print(e)
    driver.quit()
    with open(exp_file+'.break', 'w') as f:
        json.dump(full_result+merged[len(full_result):], f, indent=2)