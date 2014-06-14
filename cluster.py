from numpy import array
from nltk.cluster import KMeansClusterer, euclidean_distance
from tasks import keywords_for_query
from itertools import chain

class kw_dict(dict):
    def __init__(self, keywords):
        keywords = list(keywords)
        if len(keywords):
            max_score = keywords[0][2]
            for keyword, tag, score in keywords:
                self[keyword+'/'+tag] = score / max_score
    
    def score(self, keyword):
        return self.get(keyword, 0)

def get_keywords(url):
    keywords = keywords_for_query(url, corpus="none", analyse_pos=False)
    return kw_dict(keywords)

def vector_from_keywords(keywords, all_words):    
    return array([keywords.score(w) for w in all_words])

def vector_from_url(url, all_words):
    keywords = get_keywords(url)
    return array([keywords.score(w) for w in all_words])

def demo_2():
    from nltk.corpus import brown
    
    categories = brown.categories()[:4]
    

def demo_1():

    urls = [
        "www.ai-one.com",
        "http://en.wikipedia.org/wiki/Albert_Einstein",
        "http://en.wikipedia.org/wiki/USA",
        "http://en.wikipedia.org/wiki/Microsoft"
        ]

    keywords = [get_keywords(url) for url in urls]
    all_words = set(chain(*keywords))
    vectors = [vector_from_keywords(kw, all_words) for kw in keywords]

    clusterer = KMeansClusterer(2, euclidean_distance, repeats=10)
    clusters = clusterer.cluster(vectors, True)