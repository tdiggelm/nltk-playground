import sys
from nltk.corpus import brown
from nathan.core import Dataspace

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

if __name__ == "__main__":
    ds = Dataspace()
    import_brown(ds)