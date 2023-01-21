import spacy
import pytextrank
from fuzzywuzzy import fuzz
import json
import string
from keybert import KeyBERT
from keybert.backend._sentencetransformers import SentenceTransformerBackend
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
embedder = SentenceTransformerBackend(model)
kw_model = KeyBERT(embedder)
#kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("de_core_news_sm")
nlp_trf = spacy.load("en_core_web_trf")
#nlp_trf = spacy.load("de_dep_news_trf")
nlp.add_pipe("textrank")
nlp_trf.add_pipe("textrank")

all_stopwords = nlp.Defaults.stop_words
print('models load ready...')

def extract_keyterms(text, trf=None):
    if trf:
        doc = nlp_trf(text)
    else:
        doc = nlp(text)
    words = []
    result = []
    filtered = []

    for phrase in doc._.phrases[:15]:
        words.append(phrase.text)
        #make copy of words here
        filtered = words.copy()

    #filter out keywords that are similar
    for i, w in enumerate(words):
        for ii, ww in enumerate(words[i+1:]):
            ratio = fuzz.token_sort_ratio(w, ww)
            if ratio >= 80:
                if w not in result:
                    result.append(w)

    #compare result list to word list
    for r in result:
        for word in words:
            if r == word:
                filtered.remove(word)
                words.remove(word)
    
    #remove stop words in beginning
    new_list = []
    for i in filtered:
        phrase = i.split()
        if len(phrase) > 0:
            if phrase[0] in all_stopwords:
                new_phrase = " ".join(phrase[1:])
                new_list.append(new_phrase)
            else:
                new_list.append(i)
                
    return new_list


def keybert_extract(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=5)
    bert_keywords = []
    for k in keywords:
        if k[1] >= 0.7:
            pronoun_lst = ["i", "he", "him", "her", "it", "me", "she", "them", "they", "us", "we", "you"]
            if k[0] not in pronoun_lst:
                bert_keywords.append(k[0])
                
    #remove stop words in beginning
    new_list = []
    for i in bert_keywords:
        phrase = i.split()
        if phrase[0] in all_stopwords:
            new_phrase = " ".join(phrase[1:])
            new_list.append(new_phrase)
        else:
            new_list.append(i)
            
    return new_list


def make_keyterms(text):
    keyterms = extract_keyterms(text)
    keyterms_trf = extract_keyterms(text, trf=True)
    for k in keyterms_trf:
        if k not in keyterms:
            keyterms.append(k.strip())

    #extract with keybert
    keybert = keybert_extract(text)
    for word in keybert:
        if word not in keyterms:
            keyterms.append(word.strip())


    new_filtered = keyterms.copy()
    pronoun_lst = ["i", "he", "him", "her", "it", "me", "she", "them", "they", "us", "we", "you"]
    for f in keyterms:
        if f in string.punctuation:
            new_filtered.remove(f)
        elif len(f) == 1:
            new_filtered.remove(f)
        elif f.lower() in pronoun_lst:
            new_filtered.remove(f)
    
    #remove stop words if appear first in keyphrase
    new_list = []
    for i in new_filtered:
        phrase = i.split()
        if phrase[0] in all_stopwords:
            new_phrase = " ".join(phrase[1:])
            new_list.append(new_phrase)
        else:
            new_list.append(i)
    
    return new_list


def modify_jsons(path):
    with open(path) as file:
        l = json.load(file)
    new_json = []
    for i in tqdm(l):
        if i is None:
            new_json.append(None)
        else:
            if i['gen']['abstract']:
                keywords = make_keyterms(i['gen']['abstract'])
                i['gen']['topics'] = keywords
                assert type(keywords[0]) == str
            if i['org']['abstract']:
                keywords = make_keyterms(i['org']['abstract'])
                i['org']['topics'] = keywords
                assert type(keywords[0]) == str
            new_json.append(i)
    json.dump(new_json, open(path, 'w'), indent=2)

if __name__ == "__main__":
    modify_jsons("result/head.json")
    modify_jsons("result/tail.json")
    modify_jsons("result/middle.json")