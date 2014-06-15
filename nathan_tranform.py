from nathan.core import Dataspace as _Dataspace
import pickle

class Utf8Serializer:
    @classmethod
    def dumps(cls, string):
        return string.encode('utf-8')
    
    @classmethod    
    def loads(cls, b):
        return b.decode('utf-8')

"""
TODO:
* add this to nathan-py
* maybe check for dumps/loads functions on serializer when instanciating
* make string serializer a class accepting an optional encoding parameter
"""

class Dataspace:
    
    def __init__(
        self,
        filename=None,
        serializer=Utf8Serializer):
        
        self.serializer = serializer
        if filename is None:
            self.dataspace = _Dataspace(quants_as_bytes=True)
        else:
            self.dataspace = _Dataspace(filename, quants_as_bytes=True)
        
    def insert(self, *quants):
        
        if len(quants) == 0:
            raise ValueError("insert expects at least 1 quant")
        
        if not self.serializer is None:
            quants = list(map(self.serializer.dumps, quants))
                
        return self.dataspace.insert(quants)
        
    def fetch(self, handle):
        
        quants = self.dataspace.fetch(handle)
        
        if not self.serializer is None:
            quants = list(map(self.serializer.loads, quants))
        
        return quants
        
if __name__ == "__main__":
    ds = Dataspace()
    h = ds.insert("hello", "world")
    context = ds.fetch(h)
    print(context)
    
    ds = Dataspace(serializer=None)
    h = ds.insert(b"hello", b"world")
    context = ds.fetch(h)
    print(context)
    
    ds = Dataspace(serializer=pickle)
    h = ds.insert("hello","world",123,True,2.3,("hello","world"),{"foo":"bar"})
    context = ds.fetch(h)
    print(context)