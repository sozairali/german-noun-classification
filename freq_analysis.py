import pandas as pd
from pprint import pprint
import csv

def main():
    
    freq_nouns = pd.read_csv('subwords')
    
    freq_nouns = freq_nouns.dropna()
    
    freq_nouns = freq_nouns[:5]
    
    print(freq_nouns)
    
    headers = list(freq_nouns.columns[3:])
    
    sets_chs = []
    
    tempset = set()
    for count, header in enumerate(headers):
        for tpl in freq_nouns[header]:
            print(type(tpl))
            for l in tuple(tpl):
                print(l, sep= ", ")
                if l not in tempset:
                    tempset.add(l)
        print("") 
        print("Length of tempset:", len(tempset))
        sets_chs.append(tempset.copy()) 
        tempset.clear()
    #print(sets_chs)
                
    '''
    set_chs = set()
    
    for word in freq_nouns["Chs"]:
        #print(word)
        for w in list(word):
            if w not in set_chs:
                set_chs.add(w)
    
    print(set_chs)
    
    print(len(set_chs))
    '''
    
if __name__ == "__main__":
    main()