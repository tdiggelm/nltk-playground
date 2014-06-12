import urllib.request
import urllib.parse
import numpy as np
import base64
import json
from tasks import _fetch_url, _tokenize, keywords_for_query
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from tasks import Filter
from itertools import islice

def bing_search(query, search_type="Web", top=50):
    #search_type: Web, Image, News, Video
    key= 'xg4Hgs7RpYJLFIzsF/Hm8lPriCD7IZqiFwGv5NFEhoI'
    query = urllib.parse.quote(query)
    # create credential for authentication
    credentials = base64.b64encode(('%s:%s' % (key, key)).encode('utf-8'))
    user_agent = ('Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1;'  
        'Trident/4.0; FDM; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 1.1.4322)')
    auth = 'Basic %s' % credentials.decode("utf-8")
    base_url = 'https://api.datamarket.azure.com/Data.ashx/Bing/Search/'
    url = (base_url + search_type + '?Query=%27' + query 
        + '%27&$top=' + str(top) + '&$format=json')
    request = urllib.request.Request(url)
    request.add_header('Authorization', auth)
    request.add_header('User-Agent', user_agent)
    request_opener = urllib.request.build_opener()
    response = request_opener.open(request) 
    response_data = response.read().decode('utf-8')
    json_result = json.loads(response_data)
    return json_result['d']['results']
            
class kw_dict(dict):
    def __init__(self, keywords):
        keywords = list(keywords)
        if len(keywords):
            max_score = keywords[0][2]
            for keyword, tag, score in keywords:
                self[keyword+'/'+tag] = score / max_score
    
    def score(self, keyword):
        return self.get(keyword, 0)

def keywords_similarity(k1, k2):
    kd1 = kw_dict(k1)
    kd2 = kw_dict(k2)
    
    all_keywords = set(list(kd1.keys()) + list(kd2.keys()))
    
    v1 = [kd1.score(w) for w in all_keywords]
    v2 = [kd2.score(w) for w in all_keywords]
    
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    return np.dot(v1, v2) / n1 / n2
    
def keywords_similarity2(k1, k2):
    kd1 = kw_dict(k1)
    kd2 = kw_dict(k2)
    
    v1 = [kd1.score(w) for w in kd1.keys()]
    v2 = [kd2.score(w) for w in kd1.keys()]
    
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    return np.dot(v1, v2) / n1 / n2
    
def fetch_url(url):
    try:
        return _fetch_url(url)
    except:
        return None

def fetch_urls(urls):
    with ThreadPool(10) as pool:
        content = pool.map(lambda url: (url, fetch_url(url)), urls)
    content = dict(content)
    return content

def bing_find(keywords, corpus, preserve_entities, analyse_pos, reject_numbers, accepted_tags, reject_stopwords, reject_punctuation, top=10, similarity_scores=True):
    query = '"' + '" "'.join(item[0] for item in keywords) + '"'
    search_results = bing_search(query, top=top)
    
    urls = [item['Url'] for item in search_results]
    content = fetch_urls(urls)

    counter = 1
    for item in search_results:
        result = {
            'url': item['Url'],
            'title': item['Title'],
            'desc': item['Description'],
            'pos': counter
        }
        
        counter += 1
        
        if similarity_scores:
            text = content[result['url']]
            if not text is None:
                kw = keywords_for_query(text, corpus=corpus, preserve_entities=preserve_entities, analyse_pos=analyse_pos, fetch_urls=False)
                filtered = filter(Filter(reject_numbers, accepted_tags, reject_stopwords, reject_punctuation), kw)
                
                #filtered = list(islice(filtered, len(keywords)))
                #result['score'] = keywords_similarity(keywords, filtered)
                
                result['score'] = keywords_similarity2(keywords, filtered)
                
            else:
                result['score'] = 0
        else:
            result['score'] = 0
        
        yield result
