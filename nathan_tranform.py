from nathan.core import Dataspace as _Dataspace
import pickle

class StringSerializer:
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
    
    def dumps(self, string):
        return string.encode(self.encoding)
    
    def loads(self, b):
        return b.decode(self.encoding)

"""
TODO:
* add this to nathan-py
* maybe check for dumps/loads functions on serializer when instanciating
* implementation: maybe create _Dataspace C binding class and create python class Dataspace with StringSerializer, etc.
* call python functions dumps/loads of serializer
"""

class Dataspace:
    
    def __init__(
        self,
        filename=None,
        serializer=StringSerializer('utf-8')):
        
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