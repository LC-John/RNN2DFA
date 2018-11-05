#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:56:57 2018

@author: zhanghuangzhao
"""


import pickle
import numpy
import random

class Dataset(object):
    
    def __init__(self, path, maxlen=20):
        
        self.__maxlen = maxlen 
        
        with open(path, "rb") as f:
            d = pickle.load(f)
            self.__re = d["re"]
            self.__alphabet = d["alphabet"]
            self.__padding = len(self.__alphabet)
            self.__train_x, self.__train_y = d["train"]
            self.__test_x, self.__test_y = d["test"]
            self.__train_l = []
            self.__test_l = []
            
        self.__train_size = len(self.__train_x)
        self.__test_size = len(self.__test_x)
        
        for i in range(self.__train_size):
            self.__train_l.append(len(self.__train_x[i]))
            tmp = []
            for j in self.__train_x[i]:
                tmp.append(self.__alphabet.index(j))
            while len(tmp) < self.__maxlen:
                tmp.append(self.__padding)
            self.__train_x[i] = tmp
                
        for i in range(self.__test_size):
            self.__test_l.append(len(self.__test_x[i]))
            tmp = []
            for j in self.__test_x[i]:
                tmp.append(self.__alphabet.index(j))
            while len(tmp) < self.__maxlen:
                tmp.append(self.__padding)
            self.__test_x[i] = tmp
            
        self.__train_epoch = []
        self.__test_epoch = []
        
    def minibatch(self, batch_size):
        
        assert batch_size <= self.__train_size, "Too large batch size!"
        assert batch_size > 0, "Negative batch size!"
        if len(self.__train_epoch) < batch_size:
            self.reset_train_epoch()
        
        batch_idx = self.__train_epoch[:batch_size]
        self.__train_epoch = self.__train_epoch[batch_size:]
        x = []
        y = []
        l = []
        for i in batch_idx:
            x.append(self.__train_x[i])
            y.append(self.__train_y[i])
            l.append(self.__train_l[i])
            
        return (numpy.asarray(x, dtype=numpy.int32), 
                numpy.asarray(y, dtype=numpy.int32), 
                numpy.asarray(l, dtype=numpy.int32))
        
    def test_batch(self, batch_size):
        
        assert batch_size <= self.__test_size, "Too large batch size!"
        assert batch_size > 0, "Negative batch size!"
        if len(self.__test_epoch) < batch_size:
            self.reset_test_epoch()
        
        batch_idx = self.__test_epoch[:batch_size]
        self.__test_epoch = self.__test_epoch[batch_size:]
        x = []
        y = []
        l = []
        for i in batch_idx:
            x.append(self.__test_x[i])
            y.append(self.__test_y[i])
            l.append(self.__test_l[i])
            
        return (numpy.asarray(x, dtype=numpy.int32), 
                numpy.asarray(y, dtype=numpy.int32), 
                numpy.asarray(l, dtype=numpy.int32))
        
    def reset_train_epoch(self):
        
        self.__train_epoch = random.sample(list(range(self.__train_size)), self.__train_size)
    
    def reset_test_epoch(self):
        
        self.__test_epoch = random.sample(list(range(self.__test_size)), self.__test_size)
        
    def get_train_size(self):
        
        return self.__train_size
    
    def get_test_size(self):
        
        return self.__test_size
    
    def get_alphabet(self):
        
        return self.__alphabet
    
    def get_regular_language(self):
        
        return self.__re
        
        
        
            
            
if __name__ == "__main__":
    
    d = Dataset("tomita/tomita_3.pkl")
    