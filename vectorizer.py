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
        self.vocab = {}
        self.vocab_rev = {} if fast_revlookup else None
        for term in terms:
            self._insert(term)
    
    def _insert(self, term):
        self.vocab[term] = self.counter
        self.vocab_rev[self.counter] = term
        self.counter += 1
        
    def index(self, term):
        return self.vocab[term]
    
    def term(self, index):
        if not self.vocab_rev is None:
            return self.vocab_rev[index]
        else:
            for term, i in self.vocab.items():
                if index == i:
                    return term
            raise KeyError(index)
            
    def items(self):
        return self.vocab.items()
            
    def __contains__(self, term):
        return term in self.vocab
        
    def __iter__(self):
        return iter(self.vocab)
        
    def __getitem__(self, term):
        return self.index(term)
            
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
        
class FeatureVector:
    def __init__(self, vector, vocab):
        self.raw = vector
        self.vocab = vocab
    
    def __len__(self):
        return self.raw.shape[1]
        
    def __iter__(self):
        return enumerate(self.raw.A[0])
        
    def features(self, sort=True, reverse=True):
        v_terms = ((self.vocab.term(i), score) for i, score in self)
        if sort:
            v_terms = sorted(v_terms, key=itemgetter(1), reverse=reverse)
        return v_terms
      
"""
TODO:
    - corpus: does not offer match functionality, for this make different class (matcher) which takes a corpus as argument (corpus only returns document vectors), does NOT create matrix! lazy loading of vectors
    - matcher: different implementations (simplest one in-memory)
"""
        
class NathanCorpus:
    """
    TODO: 
    - return TagVector
    - implement compare function to compare two corpora => yields comparison matrix
    """
    
    def __init__(self, model, tags):
        self.model = model
        self.tags = Vocabulary(tags)   
        #self.ttm = vstack(self.model.vec_tag(tag).raw for tag in self.tags)
        self.ttm = lil_matrix((len(self.tags), len(self.model.vocab)))
        for tag, index in self.tags.items():
            self.ttm[index] = self.model.vec_tag(tag).raw
        
    def __iter__(self):
        return (FeatureVector(v, self.model.vocab) for v in self.ttm)
    
    def __len__(self):
        return len(self.tags)
    
    def documents(self):
        return ((tag, FeatureVector(self.ttm[index], self.model.vocab)) 
            for tag, index in self.tags.items())
        
    def _vec_to_tags(self, v, sort=True):
        v_tags = ((self.tags.term(i), val) for i, val in enumerate(v.A[0]))
        if sort:
            v_tags = sorted(v_tags, key=itemgetter(1), reverse=True)
        return v_tags
        
    def match_tag(self, tag):
        return self._vec_to_tags(self.model.vec_tag(tag).raw * self.ttm.T)
        
    def match_terms(self, *terms, how='any'):
        return self._vec_to_tags(
            self.model.vec_asso(*terms, how=how).raw * self.ttm.T)
        
    def match_text(self, text):
        return self._vec_to_tags(self.model.vec_text(text).raw * self.ttm.T)
        
    def match_url(self, url):
        return self._vec_to_tags(self.model.vec_url(url).raw * self.ttm.T)
    

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
            self.update_vocab()
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
    
    #def tags(self, prefix=None):
    #    if prefix is None:
    #        prefix = ''
    #    tags = self.model.ds.complete("@%s" % tag_prefix)
    #    return (quant[1:] for quant in tags if len(quant) > 1)
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
        if tags is None: # if no tags are specified take all
            tags = self.tags()
        return NathanCorpus(self, tags)
    
    def __iter__(self):
        return (self.vec_tag(tag) for tag in self.tags())
                    
    def update_vocab(self, vocabulary=None):
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
                ' - run update_vocab() first')
            
        vector = lil_matrix((1, len(self.vocab)), dtype=scipy.float64)
        
        for term, vic, plau in analysis:
            if term in self.vocab:
                score = vic * plau
                vector[0, self.vocab[term]] += score # update score
        
        if not self.norm is None:
            n = np.sum(np.abs(vector.A[0])**self.norm,axis=-1)**(1./self.norm)
            vector = vector / n
            
        return FeatureVector(csr_matrix(vector), self.vocab)
        
    def vec_asso(self, *terms, how='any'):
        if how == 'any': # calculate and combine assos for each search word
            asso = chain(*(self.ds.associate(term, limit=0) for term in terms))
        elif how == 'all': # all search words must be contained in assoc
            asso = self.ds.associate(*terms, limit=0)
        else:
            raise ValueError('how must be either \'any\' or \'all\'')
        return self._vectorize(asso)
    
    def vec_url(self, url):
        tmr = Timer("fetch url")
        text = _fetch_url(url)
        tmr.stop()
        
        return self.vec_text(text)
    
    def vec_text(self, text):
        tmr = Timer("tokenize text")
        sents = _tokenize(text, False, False)
        tmr.stop()
        return self.vec_sents(sents)
    
    def vec_tag(self, tag):
        handle = self.ds.select('@%s' % tag)
        if handle is None:
            raise TagNotFound(tag)
        keywords = self.ds.keywords_of(handle, limit=0)
        return self._vectorize(keywords)
        
    def vec_sents(self, sentences):
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
        nv.train_sents(reuters.sents(fileid), tags=tags)
    nv.update_vocab()
    
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
    nv.update_vocab()

"""
LSI testing:

from gensim import corpora, models, similarities
lsi = models.LsiModel(c)
index = similarities.MatrixSimilarity(lsi[corpus])
vec_lsi = lsi[model.vec_asso("oil")]
sims = index[vec_lsi]
[(c.tags.term(index), score) for index,score in sorted(list(enumerate(sims)), key=lambda i: i[1], reverse=True)[:20]]
"""