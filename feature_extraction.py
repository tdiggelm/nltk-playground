"""
TODO: 
- replace @ with # for tags
- enhance tokenizer with entity preservation
"""

from operator import itemgetter
from scipy.sparse import lil_matrix, csr_matrix
class Vocabulary:
    """Used internally for building the vocabulary"""
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
    """Exception raised when a tag does not exist."""
    pass

from bs4 import BeautifulSoup
from urllib.request import urlopen
class DefaultPreprocessor:
    """Callable, preprocess a document.
    
    Currently this default preprocessor detects if the document is a url and
    fetches its content.
    """
    def __init__(self, open_url=True):
        self.open_url = open_url
        self._is_url = re.compile(
            '^(?i)\\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+'
            '[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
            '|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)'
            '|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))$')
    
    @staticmethod
    def _decode(page):
        data = page.read()
        charset = 'utf-8'
        match = re.findall('charset=(.+)', page.info().get('Content-Type', ''))
        if (len(match) > 0):
            charset = match[0]
        data = data.decode(charset)
        return data
    
    @staticmethod
    def _fetch_url(url):
        with urlopen(url, timeout=10) as page:
            html = DefaultPreprocessor._decode(page)
        soup = BeautifulSoup(html)

        # kill all script and style elements
        for script in soup(['script', 'style']):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines 
            for phrase in line.split('  '))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    
    def __call__(self, doc):
        if (self.open_url 
                and isinstance(doc, str) 
                and not self._is_url.match(doc) is None):
            if not doc.startswith("http"):
                doc = "http://" + doc
            doc = self._fetch_url(doc)
        return doc
        
class DefaultTokenNormalizer:
    """Normalize a token (currently only tranforms to lower-case)."""
    def __call__(self, token):
        return token.lower()

from nltk import word_tokenize, sent_tokenize
class DefaultTokenizer:
    """Callable, return document tokanized into words and sentences."""
    def __call__(self, text):
        text = [word_tokenize(sent) for sent in sent_tokenize(text)]
        text = [sent for sent in text if len(sent) > 0]
        return text

import nltk
import re
class DefaultTokenFilter:
    """Callable, filter unwanted tokens (when building the vocabulary)."""
    def __init__(self, 
            reject_numbers=True, 
            reject_stopwords=True,
            reject_punctuation=True,
            language='english'):
        
        self.stopwords = set(nltk.corpus.stopwords.words(language))
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
    """The NathanModel can be used for feature extraction and vectorization.
    
    Keyword arguments:
    dataspace -- can be a filename, a nathan.core.Dataspace or None (default)
    norm -- can be 'l2', 'l1' or None (default 'l2')
    preprocessor -- callable, returns preprocessed documents
    tokenizer -- callable, returns document tokenized into words and sentences
    token_normalizer -- callable, returns normalized tokens
    token_filter -- callable, filters words when building the vocabulary
    
    Examples:
    >>> model = NathanModel("./reuters.hnn")
    """
    def __init__(self, 
            dataspace=None,
            norm="l2",
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
        self._norm = norm
        self._vocabulary = Vocabulary()
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer
        self._token_normalizer = token_normalizer
        self._token_filter = token_filter
        self._update_vocab()
    
    def _normalize(self, v):
        """Normalize vector v in-place with norm or return v if norm=None."""
        if self._norm is None:
            return v
        else:
            return sklearn_normalize(v, norm=self._norm, copy=False)
    
    def _update_vocab(self):
        """Synchronize the vocabulary with dataspace quants."""
        vocab = self._ds.all_quants(limit=0)
        vocab = (quant for quant in vocab if len(quant) > 0 and quant[0] != "@")
        vocab = filter(self._token_filter, vocab)
        self._vocabulary.fit(vocab)
    
    def _vectorize(self, analysis):
        """Turn an analysis result into a vector by using the vocab."""
        scores = ((word, vty*pty) for word, vty, pty in analysis)
        return self._normalize(self._vocabulary.transform(scores))

    def save(self, filename):
        """Save the dataspace."""
        self._ds.save(filename)

    def train(self, docs):
        """Trains the model the training documents.
        
        The docs argument is an iterator that yields documents or 
        tuples (tags, document). When using the DefaultPreprocessor, a
        document can be either a string, a url or an already tokenized
        input. The document is being linked to all the tags specified.
        
        Example:
        
        >>> docs = [
            ("doc:test", "hello world"),                    # 1
            (["doc:test2", "cat:foo"], "www.ai-one.com"),   # 2
            ("xyz", [["foo", "bar"], ["123", "yada"]]),     # 3
            "this is another document without tags"         # 4
        ]
        >>> model.train(docs)
        
        Document one is given as a raw string and tagged with "doc:test". The
        second document is tagged with "doc:test2" and "cat:foo" and fetched
        from the specified url. The third document is tagged with "xyz" and
        already given as a tokenized input (sentences and words). The last
        document is again given as a raw string while it is not tagged at all.
        """
        for item in docs:
            try:
                tags, doc = item
            except:
                tags, doc = [], item
            if isinstance(tags, str):
                tags = [tags]
            if not self._preprocessor is None:
                doc = self._preprocessor(doc)
            if not self._tokenizer is None and isinstance(doc, str):
                doc = self._tokenizer(doc)
            tag_handles = [self._ds.insert('@%s' % tag) for tag in tags]
            for sent in doc:
                if not self._token_normalizer is None:
                    sent = (self._token_normalizer(term) for term in sent)
                sen_h = self._ds.insert(sent)
                for tag_h in tag_handles:
                    self._ds.link(tag_h, sen_h)
        self._update_vocab()
    
    def transform_document(self, doc):
        """Transforms the input document into a feature vector.
        
        When using the DefaultPreprocessor a document can either be a string
        containing the raw text, a url that is then fetched by the preprocessor
        or an already tokenized input array.
        
        Examples:
        >>> model.transform_document("foo bar")
        >>> model.transform_document("http://en.wikipedia.org/wiki/Foobar")
        >>> model.transform_document([["foo", "bar"]])
        """
        if not self._preprocessor is None:
            doc = self._preprocessor(doc)
        if not self._tokenizer is None and isinstance(doc, str):
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
    
    def transform_words(self, *terms, how='any'):
        """Transforms the input words into a feature vector.
        
        Keyword arguments:
        how -- specifies how multiple words are combined (default 'any')
            'any': mean vector of all word vectors
            'all': associate the terms
            'inverse': use an inverse association scheme
        
        Examples:
        >>> model.transform_words('gold')
        >>> model.transform_words('gold', 'silver')
        >>> model.transform_words('gold', 'copper', how='all')
        """
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
        """Transforms tags into a feature vector.
        
        Examples:
        >>> model.transform_tags('cat:gold')
        >>> model.transform_tags('cat:gold', 'cat:money')
        
        If multiple tags are specified, the mean of all the vectors tranformed
        seperatly is returned.
        """
        def transform_tag(tag):
            handle = self._ds.select('@%s' % tag)
            if handle is None:
                raise TagNotFound(tag)
            keywords = self._ds.keywords_of(handle, limit=0)
            return self._vectorize(keywords)
        return self._normalize(sum(transform_tag(tag) for tag in tags))
        
    def inverse_transform(self, vector, topn=10, reverse=True):
        """Transforms a vector into (feature, score) tuples.
        
        Keywords arguments:
        topn -- Return top n features (default 10), or unsorted if None
        reverse -- Return features in reverse order
        
        Example:
        >>> vec = model.transform_words('gold')
        >>> model.inverse_transform(vec)
        [('gunnar', 0.35409381339795531),
         ('kgs', 0.28620700949846989),
         ('tyranex', 0.27177590999331386),
         ('tyranite', 0.27177590999331386),
         ('gldf', 0.20979731306159416),
         ('tunnel', 0.16583403659912016),
         ('echo', 0.15893455216408992),
         ('cove', 0.1490781458283324),
         ('mccoy', 0.1490781458283324),
         ('ngc', 0.12112428230386456)]
        """
        if not topn is None:
            ret = self._vocabulary.inverse_transform(vector, True, reverse)
            return ret[:topn]
        else:
            return self._vocabulary.inverse_transform(vector)
        
    def tags(self, pattern=None, match='glob', limit=None):
        """Return all tags in model.
        
        Keyword arguments:
        pattern -- only return tags matching the pattern (default None)
        match -- 'glob' or 'similar' (default 'glob')
            'glob': wildcard style matching, e.g. 'cat:*', 'a??le'
            'similar': find similar tags
        limit -- limit to n results
        
        Examples:
        >>> model.tags('cat:*')
        >>> model.tags('cat:gowd', match='similar')
        """
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
        """Return all words (features) in model.
        
        Keyword arguments:
        pattern -- only return words matching the pattern (default None)
        match -- 'glob' or 'similar' (default 'glob')
            'glob': wildcard style matching, e.g. 'cat:*', 'a??le'
            'similar': find similar words
        limit -- limit to n results
        
        Examples:
        >>> model.words('a??le')
        >>> model.words('helo', match='similar')
        """
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
    
    @property
    def dataspace(self):
        """Return the underlying dataspace."""
        return self._ds
    
    def __contains__(self, word):
        """Check if word is in vocabulary."""
        return word in self._vocabulary
        
    def num_features(self):
        """Return number of features, corresponds with the vector length."""
        return len(self._vocabulary)
        
    def num_tags(self):
        """Return number of tags."""
        return sum(1 for _ in self.tags())
        
    def __repr__(self):
        return ('NathanModel(num_features=%s, num_tags=%s)' 
            % (self.num_features(), self.num_tags()))
