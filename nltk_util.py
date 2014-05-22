import nltk
import re

class TextUtil:
    
    def __init__(self, text):
        self.text = text
        
    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as file:
            return cls(file.read())

    def sent_tokens(self):
        if not hasattr(self, "_sents"):
            sents = nltk.sent_tokenize(self.text)
            sents = [nltk.word_tokenize(sent) for sent in sents]
            self._sents = sents
        return self._sents
        
    def pos_tokens(self):
        if not hasattr(self, "_tagged"):
            tagged = [nltk.pos_tag(sent) for sent in self.sent_tokens()]
            self._tagged = tagged
        return self._tagged

    def named_entities(self):
        for sent in self.pos_tokens():
            for chunk in nltk.ne_chunk(sent):
                if isinstance(chunk, nltk.tree.Tree):            
                    label = chunk.label()
                    text = ' '.join(c[0] for c in chunk.leaves())
                    yield (label, text)
