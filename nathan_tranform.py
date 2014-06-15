from nathan.core import Dataspace as _Dataspace
import pickle

class Utf8Serializer:
    def dumps(self, string):
        return string.encode('utf-8')
        
    def loads(self, b):
        return b.decode('utf-8')

class Dataspace:
    
    def __init__(
        self,
        filename=None,
        serializer=Utf8Serializer()):
        
        self.serializer = serializer
        if filename is None:
            self.dataspace = _Dataspace(quants_as_bytes=True)
        else:
            self.dataspace = _Dataspace(filename, quants_as_bytes=True)
        
    def insert(self, *quants):
        
        if len(quants) == 0:
            raise ValueError("insert expects at least 1 quant")
            
        quants = list(map(self.serializer.dumps, quants))
                
        return self.dataspace.insert(quants)
        
    def fetch(self, handle):
        
        quants = self.dataspace.fetch(handle)
        
        return list(map(self.serializer.loads, quants))
        
if __name__ == "__main__":
    ds = Dataspace()
    h = ds.insert("hello", "world")
    context = ds.fetch(h)
    print(context)
    
    ds = Dataspace(serializer=pickle)
    h = ds.insert("hello","world",123,True,2.3,("hello","world"),{"foo":"bar"})
    context = ds.fetch(h)
    print(context)