# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:48:48 2018

@author: DrLC
"""

import pickle
import tomita

class Generator(object):
    
    def __init__(self, dfa):
        
        self.__dfa = dfa
        self.__Sigma = ["0", "1"]
        
    def generate(self, maxl):
        
        self.__d = [[""], [int(self.__dfa.classify(""))]]
        for l in range(1, maxl+1):
            seqs = self.__generate_seq(self.__Sigma, l)
            for s in seqs:
                self.__d[0].append(s)
                self.__d[1].append(int(self.__dfa.classify(s)))
        return self.__d
                
    def save(self, path):
        
        with open(path, "wb") as f:
            pickle.dump(self.__d, f)
        
    def __generate_seq(self, Sigma, l):
        
        if l <= 1:
            return Sigma
        lower = self.__generate_seq(Sigma, l-1)
        res = []
        for ch in Sigma:
            for s in lower:
                res.append(s+ch)
        return res
        
if __name__ == "__main__":
    
    maxl = 20
    dfas = [tomita.Tomita_1,
            tomita.Tomita_2,
            tomita.Tomita_3,
            tomita.Tomita_4,
            tomita.Tomita_5,
            tomita.Tomita_6,
            tomita.Tomita_7]
    for i in range(len(dfas)):
        G = Generator(dfas[i]())
        G.generate(maxl)
        G.save("tomita_"+str(i+1)+".pkl")
        print ("tomita "+str(i+1)+" complete!")