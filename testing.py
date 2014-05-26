import pandas as pd
import nathan.core as nc

class Dataspace:
    def __init__(self, *args, **kwargs):
        self.ds = nc.Dataspace(*args, **kwargs)
    
    def keywords_for(self, category, **kwargs):
        handle = self.ds.select("#%s" % category)
        keyw = self.ds.keywords_of(handle, **kwargs)
        columns = ("quant", "vicinity", "plausibility")
        return pd.DataFrame.from_records(keyw, columns=columns)
