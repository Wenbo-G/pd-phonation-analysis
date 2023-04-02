"""Generic helper functions to interface with pickle"""

import pandas as pd
import pickle

def pickleLoad(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data
    
def pickleSave(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f)
    
