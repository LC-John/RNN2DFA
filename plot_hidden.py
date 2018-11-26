#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:14:35 2018

@author: zhanghuangzhao
"""

import os, sys
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from classifier import SequenceClassifier
from dataset import Dataset

if __name__ == "__main__":
    
    tomita_idx = 3
    print ("TOMITA "+str(tomita_idx))
    if len(sys.argv) >= 2:
        tomita_idx = int(sys.argv[1])
    
    dataset_path = "./tomita/tomita_"+str(tomita_idx)+".pkl"
    cell_type = "gru"
    
    seq_max_len = 20
    embed_w = 5
    n_cell = 128
    
    n_epoch = 100
    batch_size = 32
    model_root = "./model/tomita_"+str(tomita_idx)+"_rnn"+"/"
    if not os.path.exists(model_root):
        os.system("mkdir "+model_root)
    model_save_path = os.path.join(model_root, "model.ckpt")
    log_save_path = "./log/tomita_"+str(tomita_idx)+"_rnn.log"
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    n_gpu = 1
    
    data = Dataset(dataset_path, seq_max_len)
    model = SequenceClassifier(seq_max_len=seq_max_len,
                               embed_w=embed_w,
                               vocab_size=len(data.get_alphabet())+1,
                               n_class=2,
                               n_hidden=n_cell,
                               cell_type=cell_type,
                               keep_prob=0.8,
                               lr=5e-4,
                               n_gpu=n_gpu,
                               grad_clip=1,
                               is_training=False)
    
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    sess.run(tf.global_variables_initializer())
    
    model.load(sess, model_save_path)
    
    for i in range(3):
        x, y, l = data.minibatch(1)
        h = model.get_hidden_states(sess, x, l)
        p, o = model.prob_op(sess, x, l)
    
        states = []
        for i in range(h.shape[0]):
            for j in range(l[i]):
                states.append(h[i][j])
        states = numpy.asarray(states)
        pca = PCA(2)
        states_2d = pca.fit_transform(states)
    
        plt.plot(states_2d[:,0], states_2d[:,1], "k-")
        plt.plot(states_2d[:1,0], states_2d[:1,1], "ro")
        if o[0] == 0:
            plt.plot(states_2d[-1:,0], states_2d[-1:,1], "go")
        else:
            plt.plot(states_2d[-1:,0], states_2d[-1:,1], "bo")
    plt.show()
    
    x, y, l = data.minibatch(100)
    h = model.get_hidden_states(sess, x, l)
    p, o = model.prob_op(sess, x, l)
    
    states = []
    for i in range(h.shape[0]):
        for j in range(l[i]):
            states.append(h[i][j])
    states = numpy.asarray(states)
    pca = PCA(2)
    states_2d = pca.fit_transform(states)
    
    plt.plot(states_2d[:,0], states_2d[:,1], ".")
    plt.show()