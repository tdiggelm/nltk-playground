from nathan.core import Dataspace
import scipy
from scipy.sparse import lil_matrix, csr_matrix, vstack
from numpy.linalg import norm
from tasks import _fetch_url, _tokenize, Timer
import string
import nltk
import re
from itertools import chain
import numpy as np
from operator import itemgetter
from itertools import islice
from fnmatch import fnmatch
    

"""
- vocab: filter out unimportant words, stopwords, hard limit, etc.
- reuters codes: http://ronaldo.cs.tcd.ie/esslli07/data/reuters21578-xml/cat-descriptions_120396.txt

mtt = nv.term_tag_matrix()
find similar tags for tag: nv.vec_to_tags(nv.vec_tag("orange")*mtt.T)[:10]
find similar tags for term: nv.vec_to_tags(nv.vec_asso("orange")*mtt.T)[:10]
find similar tags for url: nv.vec_to_tags(nv.vec_url("http://ai-one.com")*mtt.T)[:10]
"""

"""
TODO:

- add optional document name to train(self, sentences, name=None, tags=None)

- add tag vocabulary

- add document vocabulary

- create update_model() function than updates term, tag and doc dictionaries

- term_tag_matrix(self, tags=None) => return matrix class with methods for querying (e.g. similar_to_tag, similar_to_text, similar_to_sents, etc.) | if tags is specified, only those tags are used

- term_document_matrix => as above but with documents

- maintain weak-link to Vectorizer

- variant: create additional Model/Repository class that combines vectors / term-xxx-matrices instead of doing it all in Vectorizer

- discuss: find generic way to handle different tag "class" (name, category, etc.)

=> TODO:

    - create FeatureVector class
    
    - create TermTagMatrix class

"""

class TagNotFound(Exception):
    pass
    
class Vocabulary:
    def __init__(self, terms, fast_revlookup=True):
        self.counter = 0
        self.vocab = []
        self.vocab_rev = {} if fast_revlookup else None
        for term in terms:
            self._insert(term)
    
    def _insert(self, term):
        self.vocab.append(term)
        self.vocab_rev[term] = self.counter
        self.counter += 1
            
    def __contains__(self, term):
        return term in self.vocab_rev
        
    def __iter__(self):
        return iter(self.vocab)
        
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.vocab_rev[key]
        elif isinstance(key, int):
            return self.vocab[key]
        else:
            raise TypeError("key must be str or int")
            
    def __len__(self):
        return len(self.vocab)
        
    def __str__(self):
        return repr(self.vocab)
        
    def __repr__(self):
        return "Vocabulary(%s)" % str(self)

class TermFilter:    
    def __init__(self, 
            reject_numbers=True, 
            reject_stopwords=True, 
            reject_punctuation=True):
        
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.reject_numbers = reject_numbers
        self.reject_stopwords = reject_stopwords
        self.reject_punctuation = reject_punctuation
    
    def __call__(self, term):
        def isnumeric(s):
            try:
                float(s)
                return True
            except:
                return False
        
        if (self.reject_punctuation 
            and not re.match("^[\\w\\d]+[\\w\\d\\-._\\s]+$", term)):
            return False
        
        if self.reject_numbers and isnumeric(term):
            return False
                
        if self.reject_stopwords and term.lower() in self.stopwords:
            return False
            
        return True

class TermTransformer:
    def __init__(self, lower_case=True):
        self.lower_case = lower_case
    
    def __call__(self, term):
        if self.lower_case:
            term = term.lower()
        
        return term
        
class FeatureVector(lil_matrix):
    def __len__(self):
        return self.shape[1]
        
    def __iter__(self):
        return enumerate(self.A[0])
    
    def __repr__(self):
        return "FeatureVector(%s)" % repr(self.A[0])

class SimilarityMatrix(lil_matrix):    
    def __init__(self, corpus):
        lil_matrix.__init__(self, (len(corpus), len(corpus.model)))
        for index, vector in enumerate(corpus):
            if index % 10 == 0:
                print("SimilarityMatrix: inserting vector %s of %s..." % (
                    index, len(corpus)))
            self[index] = vector
    
    def __getitem__(self, key):
        if isinstance(key, FeatureVector):
            return (key * self.T).A[0]
        elif isinstance(key, SimilarityMatrix):
            return (key * self.T).A
        else:
            return lil_matrix.__getitem__(self, key)
        
    def __iter__(self):
        return ((v * self.T).A[0] for v in lil_matrix.__iter__(self))
        
    def __repr__(self):
        return "SimilarityMatrix(%s)" % repr(self.A)
        
class NathanCorpus:    
    def __init__(self, model, tags=None):
        self.model = model
        
        if tags is None: # if no tags are specified take all
            tags = self.model.tags()
        elif isinstance(tags, str): # if is string use as pattern
            tags = self.model.tags(tags)
            
        self.tags = Vocabulary(tags)
    
    def translate_similarities(self, sims, sort=True, reverse=True):
        v_tags = ((self.tags[i], val) for i, val in enumerate(sims))
        if sort:
            v_tags = sorted(v_tags, key=itemgetter(1), reverse=reverse)
        return v_tags
        
    def similarity_matrix(self):
        return SimilarityMatrix(self)
    
    def __getitem__(self, key):
        return self.tags[key]
        
    def __iter__(self):
        return (self.model.vectorize_tag(tag) for tag in self.tags)
        
    def __len__(self):
        return len(self.tags)
    
class NathanModel:
    def __init__(self, 
            filename=None, 
            term_filter=TermFilter(), 
            term_transformer=TermTransformer(),
            norm=2):
        
        self.norm = norm
        self.vocab = None
        self.term_transformer = term_transformer
        self.term_filter = term_filter
        
        if not filename is None:
            self.ds = Dataspace(filename)
            self.update_model()
        else:
            self.ds = Dataspace()
            
    def save(self, filename):
        self.ds.save(filename)
    
    def train_sents(self, sentences, tags=None):  
        tag_handles = []
        if not tags is None:
            tag_handles = [self.ds.insert('@%s' % tag) for tag in tags]
        for sent in sentences:
            if not self.term_transformer is None:
                sent = (self.term_transformer(term) for term in sent)
            sen_h = self.ds.insert(sent)
            for tag_h in tag_handles:
                self.ds.link(tag_h, sen_h)
                
    def train_text(self, text, tags=None):
        sents = _tokenize(text, False, False)
        self.train_sents(sents, tags)

    def train_url(self, url, tags=None):
        text = _fetch_url(url)
        self.train_text(text, tags)
        
    def translate_features(self, features, sort=True, reverse=True):
        v_terms = ((self.vocab[index], score) for index, score in features)
        if sort:
            v_terms = sorted(v_terms, key=itemgetter(1), reverse=reverse)
        return v_terms
    
    def tags(self, pattern=None):
        def match_pattern(quant):
            if pattern is None:
                return True
            else:
                return fnmatch(quant[1:], pattern)
            
        tags = self.ds.complete("@")
        return (quant[1:] for quant in tags 
            if len(quant) > 1 
            and match_pattern(quant))
    
    def corpus(self, tags=None):
        return NathanCorpus(self, tags)
        
    def __getitem__(self, key):
        return self.vocab[key]
    
    def __iter__(self):
        return iter(self.vocab)
        
    def __len__(self):
        return len(self.vocab)
        
    def __contains__(self, term):
        return term in self.vocab
                    
    def update_model(self, vocabulary=None):
        if not vocabulary is None:
            self.vocab = vocabulary
        else:
            quants = self.ds.all_quants(limit=0)
            filtered = filter(lambda term: term[0] != '@', quants)
            if not self.term_filter is None:
                filtered = filter(self.term_filter, filtered)
            self.vocab = Vocabulary(filtered)
    
    def _vectorize(self, analysis):
        if self.vocab is None:
            raise ValueError('vocabulary not initialized'
                ' - run update_model() first')
            
        vector = FeatureVector((1, len(self.vocab)), dtype=scipy.float64)
        
        for term, vic, plau in analysis:
            if term in self.vocab:
                score = vic * plau
                vector[0, self.vocab[term]] += score # update score
        
        if not self.norm is None:
            n = np.sum(np.abs(vector.A[0])**self.norm,axis=-1)**(1./self.norm)
            vector = FeatureVector(vector / n)
            
        return vector
    
    """
    TODO: check if term is in vocabulary, also apply input transformation to term, also do this for vectorize asso!
    """
    
    def similar(self, term, use_vocab=True, limit=0):
        if not self.term_transformer is None:
            term = self.term_transformer(term)
        it = iter(self.ds.similar_to(term))
        it = filter(lambda item: len(item[0]) > 0 and item[0][0] != '@', it)
        if use_vocab:
            it = filter(lambda item: item[0] in self.vocab, it)
        if limit > 0:
            it = islice(it, limit)
        
        term, max_score = next(it)
        yield (term, 1.0)
        
        for term, score in it:
            yield (term, score/max_score)

    def vectorize_terms(self, *terms, how='any', associate_reverse=False):
        if not self.term_transformer is None:
            terms = (self.term_transformer(term) for term in terms)
            
        if associate_reverse:
            asso = self.ds.associate_reverse(*terms, limit=0)
        elif how == 'any': # calculate and combine assos for each search word
            asso = chain(*(self.ds.associate(term, limit=0) for term in terms))
        elif how == 'all': # all search words must be contained in assoc
            asso = self.ds.associate(*terms, limit=0)
        else:
            raise ValueError('how must be either \'any\' or \'all\'')
        return self._vectorize(asso)
    
    def vectorize_url(self, url):
        tmr = Timer("fetch url")
        text = _fetch_url(url)
        tmr.stop()
        
        return self.vectorize_text(text)
    
    def vectorize_text(self, text):
        tmr = Timer("tokenize text")
        sents = _tokenize(text, False, False)
        tmr.stop()
        return self.vectorize_sents(sents)
    
    def vectorize_tag(self, tag):
        handle = self.ds.select('@%s' % tag)
        if handle is None:
            raise TagNotFound(tag)
        keywords = self.ds.keywords_of(handle, limit=0)
        return self._vectorize(keywords)
        
    def vectorize_sents(self, sentences):
        tmr = Timer("import temp text")
        doc_h = self.ds.insert('#temp')
        for sent in sentences:
            if not self.term_transformer is None:
                sent = (self.term_transformer(term) for term in sent)
            sen_h = self.ds.insert(sent)
            self.ds.link(doc_h, sen_h)
        tmr.stop()
        
        tmr = Timer("vectorize keywords")
        keywords = self.ds.keywords_of(doc_h, limit=0)
        vector = self._vectorize(keywords)
        tmr.stop()
        
        tmr = Timer("delete temp text")
        for child_h in self.ds.children_of(doc_h):
            self.ds.unlink(doc_h, child_h)
            if len(self.ds.parents_of(child_h)) == 0:
                self.ds.erase(child_h)
        self.ds.erase(doc_h)
        tmr.stop()
        
        return vector
        
def import_reuters(nv):
    from nltk.corpus import reuters
    counter = 0
    
    fileids = reuters.fileids()
    l = len(fileids)
    for fileid in fileids:
        counter += 1
        print("importing file %s of %s..." % (counter, l))
        tags = (["doc:%s" % fileid] 
            + ["cat:%s" % category for category in reuters.categories(fileid)])
        nv.train_sents(reuters.vectorize_sents(fileid), tags=tags)
    nv.update_model()
    
def import_file(nv, filename):
    counter = 0
    with open(filename) as f:
        for line in f:
            if counter % 100 == 0:
                print("importing line %s..." % counter)
            counter += 1
            line = line.strip()
            if len(line) > 0:
                nv.train_text(line)
    nv.update_model()

"""
LSI testing:

from gensim import corpora, models, similarities
lsi = models.LsiModel(corpus)
index = similarities.MatrixSimilarity(lsi[corpus])
corpus.translate(index[lsi[vectorizer.vectorize_terms("oil")]])[:10]
"""

"""
KMeans clustering:

# from each category select 10 files tops
tags = [("doc:%s" % f for f in chain(*(islice(reuters.fileids(cat), 10) for cat in reuters.categories())))]

# get corpus
corpus = model.corpus(tags)

# get similarity matrix
sim = SimilarityMatrix(corpus)

# create KMeans clusterer
from sklearn.cluster import KMeans
km = KMeans(90)

# show files, categories, prediction
clusters = [(corpus_files[index], reuters.categories(corpus_files[index][4:]),  cluster) for index, cluster in enumerate(km.fit_predict(sim.ttm))]
"""

"""
### Find similar documents for document / search terms

from vectorizer import *

model = NathanModel("./reuters-cf.hnn")

corpus = model.corpus("doc:test*")

sim = corpus.similarity_matrix() # this takes some time

[(score, fileid, reuters.raw(fileid[4:])) for fileid, score in corpus.translate_similarities(sim[model.vectorize_term("gold")])][:10]

[(score, fileid, reuters.raw(fileid[4:])) for fileid, score in corpus.translate_similarities(sim[model.vectorize_tag("doc:test/19802")])][:10]

"""