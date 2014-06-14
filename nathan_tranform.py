from nathan.core import Dataspace as _Dataspace
import pickle

class Utf8Transformer:
    def encode(self, quant):
        return quant.encode('utf-8')
        
    def decode(self, b):
        return b.decode('utf-8')
        
class PickleTransformer:
    def encode(self, quant):
        return pickle.dumps(quant)
        
    def decode(self, b):
        return pickle

class Dataspace:
    
    def __init__(
        self,
        filename=None,
        quant_tranformer=Utf8Transformer()):
        
        self.quant_transformer = quant_tranformer
        if filename is None:
            self.dataspace = _Dataspace(quants_as_bytes=True)
        else:
            self.dataspace = _Dataspace(filename, quants_as_bytes=True)
        
    def insert(self, *quants):
        
        if len(quants) == 0:
            raise ValueError("insert expects at least 1 quant")
        
        #if hasattr(quants[0], '__iter__'):
        #    quants = quants[0]
            
        quants = list(map(self.quant_transformer.encode, quants))
        
        print(quants)
        
        return self.dataspace.insert(quants)
        
    def fetch(self, handle):
        
        quants = self.dataspace.fetch(handle)
        
        return list(map(self.quant_transformer.decode, quants))
        
if __name__ == "__main__":
    ds = Dataspace()
    handle = ds.insert("hello", "world")
    context = ds.fetch(handle)
    print(context)
    