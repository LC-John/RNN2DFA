#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:19:25 2018

@author: zhanghuangzhao
"""

def write_log(msg, path):
    
    with open(path, "a") as f:
        
        f.write(msg)