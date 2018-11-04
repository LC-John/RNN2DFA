# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:43:00 2018

@author: DrLC
"""

import copy
import random

class DFAState(object):
    
    def __init__(self, idx=-1, acc=True):
        
        self.__idx = idx
        self.__acc = acc
        self.__nxt = {}
        
    def get_idx(self):
        
        return self.__idx
        
    def get_acc(self):
        
        return self.__acc
        
    def set_acc(self, acc):
        
        self.__acc = acc
        
    def get_nxt(self, key):
        
        assert key in self.__nxt.keys(), ("'%s' not in alphabet" % key)
        return self.__nxt[key]

    def set_nxt(self, key, s):
        
        self.__nxt[key] = s

class Tomita_1(object):
    
    def __init__(self):
        
        self.__re = "1*"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True), DFAState(1, False)]
        self.__n_states = 2
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[1])
        self.__states[0].set_nxt("1", self.__states[0])
        self.__states[1].set_nxt("0", self.__states[1])
        self.__states[1].set_nxt("1", self.__states[1])

    def classify(self, seq):
        
        s = self.__start
        for ch in seq:
            s = s.get_nxt(ch)
        return s.get_acc()
        
    def generate(self, l):
        
        s = self.__start
        seq = ""
        for i in range(l):
            ch = random.sample(self.__Sigma, 1)[0]
            s = s.get_nxt(ch)
            seq += ch
        return seq, s.get_acc()
        
    def get_re(self):
        
        return copy.deepcopy(self.__re)
        
    def get_alphabet(self):
        
        return copy.deepcopy(self.__Sigma)
        
class Tomita_2(object):
    
    def __init__(self):
        
        self.__re = "(10)*"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, False),
                         DFAState(2, True),
                         DFAState(3, False)]
        self.__n_states = 4
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[3])
        self.__states[0].set_nxt("1", self.__states[1])
        self.__states[1].set_nxt("0", self.__states[2])
        self.__states[1].set_nxt("1", self.__states[3])
        self.__states[2].set_nxt("0", self.__states[3])
        self.__states[2].set_nxt("1", self.__states[1])
        self.__states[3].set_nxt("0", self.__states[3])
        self.__states[3].set_nxt("1", self.__states[3])

    def classify(self, seq):
        
        s = self.__start
        for ch in seq:
            s = s.get_nxt(ch)
        return s.get_acc()
        
    def generate(self, l):
        
        s = self.__start
        seq = ""
        for i in range(l):
            ch = random.sample(self.__Sigma, 1)[0]
            s = s.get_nxt(ch)
            seq += ch
        return seq, s.get_acc()
        
    def get_re(self):
        
        return copy.deepcopy(self.__re)
        
    def get_alphabet(self):
        
        return copy.deepcopy(self.__Sigma)
        
class Tomita_3(object):
    
    def __init__(self):
        
        self.__re = "all w without containing an odd number of consecutive 0’s after an odd number of consecutive 1’s"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, True),
                         DFAState(2, False),
                         DFAState(3, True),
                         DFAState(4, False)]
        self.__n_states = 5
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[0])
        self.__states[0].set_nxt("1", self.__states[1])
        self.__states[1].set_nxt("0", self.__states[2])
        self.__states[1].set_nxt("1", self.__states[0])
        self.__states[2].set_nxt("0", self.__states[3])
        self.__states[2].set_nxt("1", self.__states[4])
        self.__states[3].set_nxt("0", self.__states[2])
        self.__states[3].set_nxt("1", self.__states[3])
        self.__states[4].set_nxt("0", self.__states[4])
        self.__states[4].set_nxt("1", self.__states[4])

    def classify(self, seq):
        
        s = self.__start
        for ch in seq:
            s = s.get_nxt(ch)
        return s.get_acc()
        
    def generate(self, l):
        
        s = self.__start
        seq = ""
        for i in range(l):
            ch = random.sample(self.__Sigma, 1)[0]
            s = s.get_nxt(ch)
            seq += ch
        return seq, s.get_acc()
        
    def get_re(self):
        
        return copy.deepcopy(self.__re)
        
    def get_alphabet(self):
        
        return copy.deepcopy(self.__Sigma)
        
class Tomita_4(object):
    
    def __init__(self):
        
        self.__re = "((1*)|(01*)|(001*))*"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, True),
                         DFAState(2, True),
                         DFAState(3, False)]
        self.__n_states = 4
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[1])
        self.__states[0].set_nxt("1", self.__states[0])
        self.__states[1].set_nxt("0", self.__states[2])
        self.__states[1].set_nxt("1", self.__states[0])
        self.__states[2].set_nxt("0", self.__states[3])
        self.__states[2].set_nxt("1", self.__states[0])
        self.__states[3].set_nxt("0", self.__states[3])
        self.__states[3].set_nxt("1", self.__states[3])

    def classify(self, seq):
        
        s = self.__start
        for ch in seq:
            s = s.get_nxt(ch)
        return s.get_acc()
        
    def generate(self, l):
        
        s = self.__start
        seq = ""
        for i in range(l):
            ch = random.sample(self.__Sigma, 1)[0]
            s = s.get_nxt(ch)
            seq += ch
        return seq, s.get_acc()
        
    def get_re(self):
        
        return copy.deepcopy(self.__re)
        
    def get_alphabet(self):
        
        return copy.deepcopy(self.__Sigma)
        
class Tomita_5(object):
    
    def __init__(self):
        
        self.__re = "all w for which the number of 0’s and 1’s are even"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, False),
                         DFAState(2, False),
                         DFAState(3, False)]
        self.__n_states = 4
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[1])
        self.__states[0].set_nxt("1", self.__states[2])
        self.__states[1].set_nxt("0", self.__states[0])
        self.__states[1].set_nxt("1", self.__states[3])
        self.__states[2].set_nxt("0", self.__states[3])
        self.__states[2].set_nxt("1", self.__states[0])
        self.__states[3].set_nxt("0", self.__states[2])
        self.__states[3].set_nxt("1", self.__states[1])

    def classify(self, seq):
        
        s = self.__start
        for ch in seq:
            s = s.get_nxt(ch)
        return s.get_acc()
        
    def generate(self, l):
        
        s = self.__start
        seq = ""
        for i in range(l):
            ch = random.sample(self.__Sigma, 1)[0]
            s = s.get_nxt(ch)
            seq += ch
        return seq, s.get_acc()
        
    def get_re(self):
        
        return copy.deepcopy(self.__re)
        
    def get_alphabet(self):
        
        return copy.deepcopy(self.__Sigma)
        
class Tomita_6(object):
    
    def __init__(self):
        
        self.__re = "all w that the difference between the numbers of 0’s and 1’s is 3n"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, False),
                         DFAState(2, False)]
        self.__n_states = 3
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[2])
        self.__states[0].set_nxt("1", self.__states[1])
        self.__states[1].set_nxt("0", self.__states[0])
        self.__states[1].set_nxt("1", self.__states[2])
        self.__states[2].set_nxt("0", self.__states[1])
        self.__states[2].set_nxt("1", self.__states[0])

    def classify(self, seq):
        
        s = self.__start
        for ch in seq:
            s = s.get_nxt(ch)
        return s.get_acc()
        
    def generate(self, l):
        
        s = self.__start
        seq = ""
        for i in range(l):
            ch = random.sample(self.__Sigma, 1)[0]
            s = s.get_nxt(ch)
            seq += ch
        return seq, s.get_acc()
        
    def get_re(self):
        
        return copy.deepcopy(self.__re)
        
    def get_alphabet(self):
        
        return copy.deepcopy(self.__Sigma)
        
class Tomita_7(object):
    
    def __init__(self):
        
        self.__re = "0*1*0*1*"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, True),
                         DFAState(2, True),
                         DFAState(3, True),
                         DFAState(4, False)]
        self.__n_states = 5
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[0])
        self.__states[0].set_nxt("1", self.__states[1])
        self.__states[1].set_nxt("0", self.__states[2])
        self.__states[1].set_nxt("1", self.__states[1])
        self.__states[2].set_nxt("0", self.__states[2])
        self.__states[2].set_nxt("1", self.__states[3])
        self.__states[3].set_nxt("0", self.__states[4])
        self.__states[3].set_nxt("1", self.__states[3])
        self.__states[4].set_nxt("0", self.__states[4])
        self.__states[4].set_nxt("1", self.__states[4])

    def classify(self, seq):
        
        s = self.__start
        for ch in seq:
            s = s.get_nxt(ch)
        return s.get_acc()
        
    def generate(self, l):
        
        s = self.__start
        seq = ""
        for i in range(l):
            ch = random.sample(self.__Sigma, 1)[0]
            s = s.get_nxt(ch)
            seq += ch
        return seq, s.get_acc()
        
    def get_re(self):
        
        return copy.deepcopy(self.__re)
        
    def get_alphabet(self):
        
        return copy.deepcopy(self.__Sigma)
        
if __name__ == "__main__":
    
    G1 = Tomita_1()
    G2 = Tomita_2()
    G3 = Tomita_3()
    G4 = Tomita_4()
    G5 = Tomita_5()
    G6 = Tomita_6()
    G7 = Tomita_7()