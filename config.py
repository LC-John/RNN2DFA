#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:51:48 2018

@author: zhanghuangzhao
"""

import tensorflow as tf

tf.app.flags.DEFINE_integer("tomita", 1,
                            "Index of the regular grammar in Tomita grammars")
tf.app.flags.DEFINE_string("dataset_root", "./tomita/",
                           "Directory where all datasets of tomita grammars are stored")
tf.app.flags.DEFINE_string("model_root", "./model/",
                           "Directory where all RNN models are stored")
tf.app.flags.DEFINE_string("log_root", "./log/",
                           "Directory where all training logs are stored")

tf.app.flags.DEFINE_string("cell", "rnn",
                           "Cell type (rnn/gru/lstm) of the RNN")
tf.app.flags.DEFINE_integer("max_seq_len", 20,
                            "Max length of the sequences")
tf.app.flags.DEFINE_integer("embed_w", 5,
                            "Width of the embedding vectors")
tf.app.flags.DEFINE_integer("cell_n", 128,
                            "cell number of the single RNN layer")

tf.app.flags.DEFINE_string("gpu", "0,1,2,3",
                           "Selection of GPU('s)")

tf.app.flags.DEFINE_boolean("is_training", True,
                            "Train the model, or load and use it")

tf.app.flags.DEFINE_integer("epoch_n", 100,
                            "Number of training epoches")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Size of each minibatch")
tf.app.flags.DEFINE_float("learning_rate", 1e-5,
                          "Learning rate during training")
tf.app.flags.DEFINE_float("keep_prob", 0.8,
                          "Probability of keeping when applying drop-out")
tf.app.flags.DEFINE_boolean("stdout", True,
                            "Use STDOUT instead of redirecting it to log")
tf.app.flags.DEFINE_boolean("stderr", True,
                            "Use STDERR instead of redirecting it to log")