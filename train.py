import sys
import nltk
from nltk.corpus import brown
from nltk.corpus import reuters
from nathan.core import Dataspace

def simplify_tag(tup):
    word, tag = tup
    try:
        tag = nltk.tag.mapping.map_tag('en-ptb', 'universal', tag)
    except:
        tag = "X"
    return (word, tag)

def import_brown(ds, silent=False, log=sys.stdout):
    """
    Import the brown corpus into `ds`. E.g.
    
    >>> from nathan.core import Dataspace
    >>> ds = Dataspace()
    >>> %time brown.import_brown(ds, silent=True)
    CPU times: user 12min 28s, sys: 536 ms, total: 12min 29s
    Wall time: 12min 29s
    """
    if not silent:
        total = len(brown.sents())
        counter = 0
    for category in brown.categories():
        cat_handle = ds.insert("#%s" % category)
        for sent in brown.sents(categories=category):
            norm = [word.lower() for word in sent]
            sen_handle = ds.insert(norm)
            ds.link(cat_handle, sen_handle)
            if not silent:
                counter += 1
                if (counter % 100 == 0):
                    print("importing %s of %s sentences..." % (counter, total), 
                        file=log)
                        
def import_brown_pos(ds, simplify_tags=False, silent=False, log=sys.stdout):
    """
    Import the brown corpus into `ds`. E.g.
    
    >>> from nathan.core import Dataspace
    >>> ds = Dataspace()
    >>> %time brown.import_brown(ds, silent=True)
    CPU times: user 12min 28s, sys: 536 ms, total: 12min 29s
    Wall time: 12min 29s
    """
    if not silent:
        total = len(brown.sents())
        counter = 0
    for category in brown.categories():
        cat_handle = ds.insert("#%s" % category)
        for sent in brown.tagged_sents(categories=category):
            if simplify_tags:
                norm = (simplify_tag(t) for t in sent)
            norm = [nltk.tuple2str(t) for t in norm]
            sen_handle = ds.insert(norm)
            ds.link(cat_handle, sen_handle)
            if not silent:
                counter += 1
                if (counter % 100 == 0):
                    print("importing %s of %s sentences..." % (counter, total), 
                        file=log)
                        
def import_reuters_flat(ds, silent=False, log=sys.stdout):
    """
    Import the brown corpus into `ds`. E.g.
    
    >>> from nathan.core import Dataspace
    >>> ds = Dataspace()
    >>> %time brown.import_brown(ds, silent=True)
    CPU times: user 12min 28s, sys: 536 ms, total: 12min 29s
    Wall time: 12min 29s
    """
    if not silent:
        total = len(reuters.sents())
        counter = 0
    root_handle = ds.insert("#reuters")
    for sent in reuters.sents():
        norm = [word.lower() for word in sent]
        sen_handle = ds.insert(norm)
        ds.link(root_handle, sen_handle)
        if not silent:
            counter += 1
            if (counter % 100 == 0):
                print("importing %s of %s sentences..." % (counter, total), 
                    file=log)

def import_reuters_flat_pos(ds, silent=False, log=sys.stdout):
    """
    Import the brown corpus into `ds`. E.g.
    
    >>> from nathan.core import Dataspace
    >>> ds = Dataspace()
    >>> %time brown.import_brown(ds, silent=True)
    CPU times: user 12min 28s, sys: 536 ms, total: 12min 29s
    Wall time: 12min 29s
    """
    
    tagger = nltk.data.load("./models/treebank_brill_aubt/treebank_brill_aubt.pickle")
    
    if not silent:
        total = len(reuters.sents())
        counter = 0
    root_handle = ds.insert("#reuters")
    for sent in reuters.sents():
        sent = tagger.tag(sent)
        norm = [nltk.tuple2str(t) for t in sent]
        sen_handle = ds.insert(norm)
        ds.link(root_handle, sen_handle)
        if not silent:
            counter += 1
            if (counter % 100 == 0):
                print("importing %s of %s sentences..." % (counter, total), 
                    file=log)
        
def import_reuters_files(ds, silent=False, log=sys.stdout):
    """
    Import the brown corpus into `ds`. E.g.
    
    >>> from nathan.core import Dataspace
    >>> ds = Dataspace()
    >>> %time brown.import_brown(ds, silent=True)
    CPU times: user 12min 28s, sys: 536 ms, total: 12min 29s
    Wall time: 12min 29s
    """
    if not silent:
        total = len(reuters.fileids())
        counter = 0
    root_handle = ds.insert("#reuters")
    for fileid in reuters.fileids():
        tags = ["@%s" % category for category in reuters.categories(fileid)]
        file_handle = ds.insert(["#%s" % fileid] + tags)
        ds.link(root_handle, file_handle)
        for sent in reuters.sents(fileid):
            norm = [word.lower() for word in sent]
            sen_handle = ds.insert(norm)
            ds.link(file_handle, sen_handle)
        if not silent:
            counter += 1
            if (counter % 10 == 0):
                print("importing %s of %s files..." % (counter, total), 
                    file=log)

# TODO: create functions import_all_seperate import_all_one

if __name__ == "__main__":
    ds = Dataspace()
    import_brown(ds)