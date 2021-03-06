"""
ds = Dataspace()

trainer = Trainer(ds)
trainer.fit_sents([["hello", "world", ...], ...], tags=["cat:foo", "doc:bar"])
...

corpus = Corpus(ds, 'cat:*')
index = MatrixSimilarity(corpus)
search = corpus.transform_words('metal', 'shiny')
matches = index[search]
top10 = corpus.tags.sorted(matches)[:10]

index_vocab = MatrixSimilarity(corpus.words) => returns matrix that contains the word vector for each word associated, can be used for finding similar words etc. (e.g. create weighted average vector and use as query)

corpus.tags & corpus.words => Dictionary
Dictionary[pos] => string
Dictionary[string] => pos
Dictionary[list of values] => list of tuples (string, value)
Dictionary.sorted(list of values, reverse=True) => sorted(self[matches], key=itemgetter(1), reverse=reverse)

=> abstrakt: handle numbers => corpus
             words => vocab
             

Dictionary.__init__(iterator) => Dictionary instance (iterator yields words)
Dictionary[Analysis] => Vector
Dictionary[Similarities] => Vector
Dictionary[Vector] => iterator of 2-tuples word, score
Dictionary[index] => word
Dictionary[word] => int
Dictionary.translate(Vector)


ds = Dataspace()

vocab = Dictionary(quant for quant in ds.all_quants() if len(quant) > 2)
for handle in ds.children_of(root)
    features = vocab[word, sim*vic for word, sim, vic in ds.keywords_of(10)]

"""

from numpy import float32
from scipy.sparse import lil_matrix
class Vector:
    def __init__(self, arg):
        if isinstance(arg, int):
            self._mat = lil_matrix((1, arg), dtype=float32)
        else:
            self._mat = lil_matrix((1, len(arg)), dtype=float32)
            for index, value in enumerate(arg):
                self[index] = value
        
    def __setitem__(self, key, value):
        self._mat[0, key] = value
        
    def __getitem__(self, key):
        return self._mat[0, key]
        
    def __len__(self):
        return self._mat.shape[1]
        
    def __iter__(self):
        return iter(self._mat.A[0])
    
    def toarray(self):
        return self.A
        
    def tosparse(self):
        return self.M
        
    @property
    def A(self):
        return self._mat.A[0]
        
    @property
    def M(self):
        return self._mat
    
    def __repr__(self):
        return "Vector(%s)" % str(self.toarray())
        
from operator import itemgetter
class Dictionary:
    def __init__(self, words):
        self._counter = 0
        self._vocab = []
        self._vocabrev = {}
        for word in words:
            self._insert(word)
    
    def _insert(self, word):
        self._vocab.append(word)
        self._vocabrev[word] = self._counter
        self._counter += 1
    
    def translate(self, vector):    
        return ((self._vocab[i], score) for i, score in enumerate(vector))
        
    def translate_sorted(self, vector, reverse=True):
        return sorted(self.translate(vector), 
            key=itemgetter(1), reverse=reverse)
        
    def vectorize(self, items):
        vector = Vector(len(self))
        for word, value in items:
            vector[self._vocabrev[word]] = value
        return vector
    
    def __contains__(self, word):
        return word in self._vocabrev
        
    def __iter__(self):
        return iter(self._vocab)
        
    def __getitem__(self, selector):
        if isinstance(selector, str):
            return self._vocabrev[selector]
        elif isinstance(selector, int):
            return self._vocab[selector]
        elif isinstance(selector, Vector):
            return self.translate(selector)
        else:
            return self.vectorize(selector)
            
    def __len__(self):
        return len(self._vocab)
        
    def __str__(self):
        return repr(self._vocab)
        
    def __repr__(self):
        return "Dictionary(%s)" % str(self)

from operator import itemgetter
from scipy.sparse import lil_matrix, csr_matrix
class DictionaryVectorizer:
    def __init__(self, vocab=None):
        if not vocab is None:
            self.fit(vocab)
            
    def fit(self, vocab):
        self._vocab = []
        self._vocabrev = {}
        for index, word in enumerate(vocab):
            self._vocab.append(word)
            self._vocabrev[word] = index
    
    def inverse_transform(self, vector, sort=False, reverse=False):    
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
        return "DictionaryVectorizer(%s)" % str(self._vocab)


"""
train = NathanTrainer(ds)
train.fit(sents, tags)

also check out http://scikit-learn.org/stable/modules/feature_extraction.html#sparsity
for how the feature extraction is organised

=> idea:
fit([
    {tags: ["doc:001", "cat:foo"], sents: [["...", ...], [...]]}
    {tags: ["doc:002", "cat:bar"], sents: [["...", ...], [...]]}
    ...
]) => automatically updates vocabulary

# model = NathanModel(ds, vocabulary=None) => is this the Vocabulary????
model = NathanModel(ds, vocab_filter, vocab_transformer)
model.fit_sents(sents, tags)
model.fit_text(text, tags)
model.fit_url(url, tags)
model.update()
model.__iter__() => vocab
model.tags([pattern]) => returns tags
model.transform_tag(sentences) => vector
model.transform_terms(word0, word1, ..., how='all'|'any'|'reverse')
model.transform_sents(sentences)
model.transform_text(text)
model.transform_url(url)
model.associate(term0, ...)
model.associate_reverse(term0, term1, ...)
model.similar_to(term)

tsm = TagSimilarityMatrix(model, pattern)
tsm.match_text() => matching tags
tsm.match_terms() => matching tags
tsm.match_url() => matching tags
tsm.match_sents() => matching tags

tsm = TermSimilarityMatrix(model, pattern)
tsm.find_similar(positive, negative)
tsm.doesnt_match(term0, term1, term3, ...)
tsm.similarity(term0, term1)

oder:

model.update(vocabulary=None) # updates vocabulary
model.vocabulary = xxx # set vocab manually

=> GENERIC

from gensim.matutils import scipy2sparse
model = Model(filename=None, term_filter=None, norm='l2')
model.train(sents, tags)
model.train(sents, tags)
model.train(sents, tags)
...
model.update() => updates vocabulary
model.vocabulary => Vectorizer
model.complete() # filtered by vocabulary
model.similar_to() # filtered by vocabulary
model.transform_sents(sents)
model.transform_tag(tag)
model.transform_terms(term0, term1, ..., how='all|any|inverse')
tags = list(model.tags("cat:*"))
corpus = list(scipy2sparse(model.transform_tag(tag)) for tag in tags)
lsi = LsiModel(corpus, num_topics=80, chunksize=2000)
index = MatrixSimilarity(lsi[corpus])
sims = index[lsi[scipy2sparse(model.transform_terms("hello"))]]
result = sorted(zip(tags, sims), reverse=True)

=> another approach

trainer = Trainer(ds)
trainer.fit(sents, tags)
trainer.fit(sents, tags)
trainer.fit(sents, tags)
...

model = Model(ds, vocabulary=None) # if vocabulary=None => use all_quants
model.transform_sents(sents)
model.transform_tag(sents)
model.transform_tag(sents)

=> best to not enforcing a workflow, build all components seperately with loose coupling

=> important: when doing similarity analyis (e.g. dot multiplication) always compute: (M * v.T).T to get the result, v * M yields the wrong result!


"""
