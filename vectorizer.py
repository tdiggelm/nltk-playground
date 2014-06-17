from nathan.core import Dataspace
import scipy
from scipy.sparse import lil_matrix, csr_matrix
from numpy.linalg import norm
from tasks import _fetch_url, _tokenize, Timer
import string
import nltk
import re
from operator import itemgetter

"""
- vocab: filter out unimportant words, stopwords, hard limit, etc.
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

class NathanVectorizer:
    
    def __init__(self, 
            filename=None, 
            term_filter=TermFilter(), 
            term_transformer=TermTransformer()):
        
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
        
    def tags(self):
        return (term[1:] for term in self.ds.complete("@") if len(term) > 1)
                    
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
                vector[0, self.vocab[term]] = vic * plau
            
        vector = vector / norm(vector.A)
            
        return csr_matrix(vector)
        
    def vec_to_terms(self, v, sort=True):
        v_terms = ((self.vocab.term(i), val) for i, val in enumerate(v.A[0]))
        if sort:
            v_terms = sorted(v_terms, key=itemgetter(1), reverse=True)
        return v_terms
        
    def vec_asso(self, *terms):
        asso = self.ds.associate(*terms, limit=0)
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
        
def demo():
    nv = NathanVectorizer()
    nv.train_sents([['hello', 'world'], ['foo', 'bar']], tags=['test'])
    nv.update_vocab()
    return nv.vec_tag('test')