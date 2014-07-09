"""
TODO: replace @ with # for tags
"""

from operator import itemgetter
from scipy.sparse import lil_matrix, csr_matrix
class Vocabulary:
    def __init__(self, vocab=None):
        self.fit(vocab)
            
    def fit(self, vocab):
        self._vocab = []
        self._vocabrev = {}
        if not vocab is None:
            for index, word in enumerate(vocab):
                self._vocab.append(word)
                self._vocabrev[word] = index
    
    def inverse_transform(self, vector, sort=False, reverse=True):    
        ret = ((self._vocab[i], score) for i, score in enumerate(vector.A[0]))
        if sort:
            ret = sorted(ret, key=itemgetter(1), reverse=reverse)
        return ret
        
    def fit_transform(self, items, ignore_missing=True):
        self.fit(term for term, _ in items)
        return self.transform(items, ignore_missing)

    def transform(self, items, ignore_missing=True):
        lil = lil_matrix((1, len(self._vocab)))
        for word, value in items:
            try:
                lil[0, self._vocabrev[word]] = value
            except KeyError as e:
                if not ignore_missing:
                    raise e
        return csr_matrix(lil)
    
    def __contains__(self, word):
        return word in self._vocabrev
        
    def __iter__(self):
        return iter(self._vocab)
        
    def __getitem__(self, selector):
        if isinstance(selector, str):
            return self._vocabrev[selector]
        elif isinstance(selector, int):
            return self._vocab[selector]
        elif isinstance(selector, csr_matrix):
            return self.inverse_transform(selector)
        else:
            return self.transform(selector)
            
    def __len__(self):
        return len(self._vocab)
        
    def __repr__(self):
        return 'Vocabulary(%s)' % str(self._vocab)

class TagNotFound(Exception):
    pass

class DefaultPreprocessor:
    def __call__(self, doc):            
        return doc
        
class DefaultTokenNormalizer:
    def __call__(self, token):
        return token.lower()

from nltk import word_tokenize, sent_tokenize
class DefaultTokenizer:
    def __call__(self, text):
        if isinstance(text, str):
            text = [word_tokenize(sent) for sent in sent_tokenize(text)]
            text = [sent for sent in text if len(sent) > 0]
        return text

import nltk
import re
class DefaultTokenFilter:    
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

from nathan.core import Dataspace
from sklearn.preprocessing import normalize as sklearn_normalize
from fnmatch import fnmatch
from itertools import islice
class NathanModel:
    def __init__(self, 
            dataspace=None,
            norm='l2',
            preprocessor=DefaultPreprocessor(),
            tokenizer=DefaultTokenizer(),
            token_normalizer=DefaultTokenNormalizer(),
            token_filter=DefaultTokenFilter()):
        if dataspace is None:
            self._ds = Dataspace()
        elif isinstance(dataspace, str):
            self._ds = Dataspace(dataspace)
        elif isinstance(dataspace, Dataspace):
            self._ds = dataspace
        else:
            TypeError("dataspace must be str, nathan.core.Dataspace or None")
        self._vocabulary = Vocabulary()
        self._norm = norm
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer
        self._token_normalizer = token_normalizer
        self._token_filter = token_filter
        self._update_vocab()
    
    def _normalize(self, v):
        return sklearn_normalize(v)
    
    def _update_vocab(self):
        vocab = self._ds.all_quants(limit=0)
        vocab = (quant for quant in vocab if len(quant) > 0 and quant[0] != "@")
        vocab = filter(self._token_filter, vocab)
        self._vocabulary.fit(vocab)
    
    def _vectorize(self, analysis):
        scores = ((word, vty*pty) for word, vty, pty in analysis)
        return self._normalize(self._vocabulary.transform(scores))

    def save(self, filename):
        self._ds.save(filename)

    def train(self, docs):
        for item in docs:
            try:
                tags, doc = item
            except:
                tags, doc = [], item
            if isinstance(tags, str):
                tags = [tags]
            if not self._preprocessor is None:
                doc = self._preprocessor(doc)
            if not self._tokenizer is None:
                doc = self._tokenizer(doc)
            tag_handles = [self._ds.insert('@%s' % tag) for tag in tags]
            for sent in doc:
                if not self._token_normalizer is None:
                    sent = (self._token_normalizer(term) for term in sent)
                sen_h = self._ds.insert(sent)
                for tag_h in tag_handles:
                    self._ds.link(tag_h, sen_h)
        self._update_vocab()
    
    def transform_doc(self, doc):
        if not self._preprocessor is None:
            doc = self._preprocessor(doc)
        if not self._tokenizer is None:
            doc = self._tokenizer(doc)
            
        doc_h = self._ds.insert('#temp')
        for sent in doc:
            if not self._token_normalizer is None:
                sent = (self._token_normalizer(term) for term in sent)
            sen_h = self._ds.insert(sent)
            self._ds.link(doc_h, sen_h)
        
        keywords = self._ds.keywords_of(doc_h, limit=0)
        
        for child_h in self._ds.children_of(doc_h):
            self._ds.unlink(doc_h, child_h)
            if len(self._ds.parents_of(child_h)) == 0:
                self._ds.erase(child_h)
        self._ds.erase(doc_h)
        
        return self._vectorize(keywords)
    
    def transform_terms(self, *terms, how='any'):
        if not self._token_normalizer is None:
            terms = (self._token_normalizer(term) for term in terms)
            
        if how == 'inverse':
            asso = self._ds.associate_reverse(*terms, limit=0)
            return self._vectorize(asso)
        elif how == 'all': # all search words must be contained in assoc
            asso = self._ds.associate(*terms, limit=0)
            return self._vectorize(asso)
        elif how == 'any': # calculate and combine assos for each search word
            assos = (self._ds.associate(term, limit=0) for term in terms)
            assos = map(self._vectorize, assos)
            return self._normalize(sum(assos))
        else:
            raise ValueError('how must be either \'any\' or \'all\'')
        
    def transform_tags(self, *tags):
        def transform_tag(tag):
            handle = self._ds.select('@%s' % tag)
            if handle is None:
                raise TagNotFound(tag)
            keywords = self._ds.keywords_of(handle, limit=0)
            return self._vectorize(keywords)
        return self._normalize(sum(transform_tag(tag) for tag in tags))
        
    def inverse_transform(self, vector, topn=10, reverse=True):
        if not topn is None:
            ret = self._vocabulary.inverse_transform(vector, True, reverse)
            return ret[:topn]
        else:
            return self._vocabulary.inverse_transform(vector)
        
    def tags(self, pattern=None, match='glob', limit=None):
        if match not in ['similar', 'glob']:
            raise ValueError("match must be 'glob' or 'similar'")
          
        if pattern == None:
            tags = self._ds.complete("@")
            tags = (tag[1:] for tag in tags if len(tag) > 1)
        elif match == 'glob':
            tags = self._ds.complete("@")
            tags = (quant[1:] for quant in tags 
                if len(quant) > 1 
                and fnmatch(quant[1:], pattern))
        elif match == 'similar':
            tags = self._ds.similar_to(pattern)
            tags = (tag[1:] for tag, _ in tags 
                if len(tag) > 1 
                and tag[0] == '@')
        else:
            raise ValueError()
        
        if not limit is None:
            tags = islice(tags, limit)
        
        return tags
    
    def words(self, pattern=None, match='glob', limit=None):
        if match not in ['similar', 'glob']:
            raise ValueError("match must be 'glob' or 'similar'")

        if pattern == None or match == 'glob':
            words = (word for word in self._vocabulary)
            if not pattern is None:
                words = (word for word in words if fnmatch(word, pattern))
        elif match == 'similar':
            words = self._ds.similar_to(pattern, limit=0)
            words = (word for word, _ in words if word in self._vocabulary)
        else:
            raise ValueError()
        
        if not limit is None:
            words = islice(words, limit)
        
        return words
        
    def __contains__(self, word):
        return word in self._vocabulary
        
    def __len__(self):
        return len(self._vocabulary)
        
    def __repr__(self):
        return ('NathanModel(tags=%s, features=%s)' 
            % (len(list(self.tags())), len(self._vocabulary)))
        
def test():
    model = NathanModel()
    model.train([("test", "hello world"), ("test2", "foo bar")])
    print(model.transform_tags("test"))
    return model