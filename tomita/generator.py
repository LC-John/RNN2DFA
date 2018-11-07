# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:48:48 2018

@author: DrLC
"""

import pickle
import numpy

import tomita

class Generator(object):
    
    def __init__(self, dfa, pass_rate, train_ratio=0.5):
        
        self.__dfa = dfa
        self.__Sigma = ["0", "1"]
        if train_ratio <= 0:
            self.__train_ratio = 0
        elif train_ratio >= 1:
            self.__train_ratio = 1
        else:
            self.__train_ratio = train_ratio
        if pass_rate <= 0:
            self.__pass_rate = 0
        elif pass_rate >= 1:
            self.__pass_rate = 1
        else:
            self.__pass_rate = pass_rate
        
    def generate(self, maxl):
        
        self.__d = [[""], [int(self.__dfa.classify(""))]]
        for l in range(1, maxl+1):
            seqs = self.__generate_seq(self.__Sigma, l)
            for s in seqs:
                if self.__pass_rate >= numpy.random.uniform(0, 1):
                    continue
                self.__d[0].append(s)
                self.__d[1].append(int(self.__dfa.classify(s)))
        print ("N = "+str(len(self.__d[0])))
        split_idx = int(self.__train_ratio * len(self.__d[0]))
        self.__d_te = [self.__d[0][split_idx:], self.__d[1][split_idx:]]
        self.__d_tr = [self.__d[0][:split_idx], self.__d[1][:split_idx]]

    def save(self, path):
        
        with open(path, "wb") as f:
            pickle.dump({"re": self.__dfa.get_re(),
                         "alphabet": self.__dfa.get_alphabet(),
                         "train": self.__d_tr,
                         "test": self.__d_te}, f)
        
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
        G = Generator(dfas[i](), 0.5, 0.5)
        G.generate(maxl)
        G.save("tomita_"+str(i+1)+".pkl")
        print ("tomita "+str(i+1)+" complete!")