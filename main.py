#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:42:24 2018

@author: zhanghuangzhao
"""

import tensorflow as tf

import config

flags = tf.app.flags.FLAGS

def main(_):
    
    is_training = flags.is_training
    tomita_idx = flags.tomita
    dataset_path = flags.dataset_root"./tomita/tomita_"+str(tomita_idx)+".pkl"
    
    if is_training: # train
        pass
    else:
        pass

if __name__ == "__main__":
    
    tf.app.run(main=main)