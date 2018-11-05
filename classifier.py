#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:37:25 2018

@author: zhanghuangzhao
"""

import tensorflow as tf
import numpy
import os

import dataset
from util import write_log

class SequenceClassifier(object):
    
    def __init__(self, seq_max_len=20, embed_w=300, vocab_size=30000, n_class=2,
                 n_layer=2, n_hidden=300, keep_prob=0.9, lr=1e-5, is_training=True):
        
        self.__is_training = is_training
        if is_training == False:
            is_training = True
            keep_prob = 1.0
        
        # placeholders for the language model -- input, output & valid seq length
        self.__X = tf.placeholder(shape=[None, seq_max_len],
                                  dtype=tf.int32, name="input")
        self.__Y = tf.placeholder(shape=[None, ],
                                  dtype=tf.int32, name="label")
        self.__L = tf.placeholder(shape=[None], dtype=tf.int32, name="valid_len")
        
        # embedding
        with tf.device("/cpu:0"):
            self.__embed_matrix = tf.get_variable("embedding_martix",
                                                  [vocab_size, embed_w],
                                                  dtype=tf.float32)
            self.__embed = tf.nn.embedding_lookup(self.__embed_matrix, self.__X,
                                                  name="embedding")
        # input dropout
        if is_training and keep_prob < 1:
            self.__rnn_in = tf.nn.dropout(self.__embed, keep_prob,
                                          name="input_dropout")
        else:
            self.__rnn_in = self.__embed
        
        # rnn architecture
        self.__cell_list_fw = [self.__make_cell(is_training, n_hidden, keep_prob, i)
                                for i in range(n_layer)]
        self.__cell_fw = tf.contrib.rnn.MultiRNNCell(self.__cell_list_fw)
        self.__cell_list_bw = [self.__make_cell(is_training, n_hidden, keep_prob, i)
                                for i in range(n_layer)]
        self.__cell_bw = tf.contrib.rnn.MultiRNNCell(self.__cell_list_bw)
        self.__rnn_outs, self.__rnn_states = tf.nn.bidirectional_dynamic_rnn(self.__cell_fw,
                                                                             self.__cell_bw,
                                                                             self.__rnn_in,
                                                                             self.__L,
                                                                             dtype=tf.float32)
        self.__rnn_out_fw = tf.reduce_mean(self.__rnn_outs[0], axis=1, name="rnn_outputs_fw")
        self.__rnn_out_bw = tf.reduce_mean(self.__rnn_outs[1], axis=1, name="rnn_outputs_bw")
        
        self.__rnn_out = tf.concat((self.__rnn_out_fw, self.__rnn_out_bw),
                                     -1, name="rnn_outputs_concat")
        
        # softmax output
        self.__dense_W = tf.Variable(tf.random_normal([2*n_hidden, n_class]),
                                     name="dense_w")
        self.__dense_b = tf.Variable(tf.constant(0, dtype=tf.float32,
                                                 shape=[n_class]),
                                     name="dense_b")
        self.__logit = tf.matmul(self.__rnn_out, self.__dense_W) + self.__dense_b
        self.__prob = tf.nn.softmax(self.__logit, name="probability")

        # cross entropy loss
        self.__Y_onehot = tf.one_hot(self.__Y, n_class)
        self.__loss_vec = tf.nn.softmax_cross_entropy_with_logits(labels=self.__Y_onehot,
                                                                  logits=self.__logit,
                                                                  name="loss_for_each")
        self.__loss = tf.reduce_mean(self.__loss_vec, name="loss")
        
        # accuracy
        self.__output = tf.cast(tf.argmax(self.__prob, -1,
                                          name="output_label"), tf.int32)
        self.__accurate = tf.equal(self.__Y, self.__output, name="accurate_output")
        self.__accuracy = tf.reduce_mean(tf.cast(self.__accurate, tf.float32),
                                         name="accuracy")
        
        # training operation
        if is_training:
            self.__opt = tf.train.AdamOptimizer(lr, name="adam_optimizer")
            self.__train_op = self.__opt.minimize(self.__loss,
                                                  global_step=tf.train.get_or_create_global_step())

        self.__saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

    def train_op(self, sess, X, Y, L):
        
        if self.__is_training:
            _, l, a = sess.run((self.__train_op, self.__loss, self.__accuracy),
                               feed_dict={self.__X: X,
                                          self.__Y: Y,
                                          self.__L: L})
            return l, a
        else:
            return None
    
    def test_op(self, sess, X, Y, L):
        
        l, a = sess.run((self.__loss, self.__accuracy),
                        feed_dict={self.__X: X,
                                   self.__Y: Y,
                                   self.__L: L})
        return l, a
    
    def prob_op(self, sess, X, L):
        
        p, o = sess.run((self.__prob, self.__output),
                        feed_dict={self.__X: X,
                                   self.__L: L})
        return p, o

    def save(self, sess, path):
        
        self.__saver.save(sess, path)
        
    def load(self, sess, path):
        
        self.__saver.restore(sess, path)

    def __make_cell(self, is_training, n_hidden, keep_prob, layer=0):
        
        # make the basic LSTM cells
        tmp_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0,
                                                reuse=not is_training)
        # dropout if needed
        if is_training and keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(tmp_cell,
                                                 output_keep_prob=keep_prob)
        else:
            cell = tmp_cell
        return cell
    
if __name__ == "__main__":
        
    seq_max_len = 200
    vocab_size = 30000
    embed_w = 300
    n_layer = 2
    n_cell = 128
    
    n_epoch = 20
    batch_size = 32
    model_save_path = "./model/imdb_classifier/model.ckpt"
    log_save_path = "./log/imdb_classifier.log"
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    
    tr = dataset.SeqClassificationDataset("./imdb/train_input.pkl",
                                          "./imdb/train_output.pkl",
                                          "./imdb/vocab.pkl",
                                          seq_max_len, vocab_size, 1.0)
    te = dataset.SeqClassificationDataset("./imdb/test_input.pkl",
                                          "./imdb/test_output.pkl",
                                          "./imdb/vocab.pkl",
                                          seq_max_len, vocab_size, 0.0)
    assert tr.get_dict_size() == te.get_dict_size()
    m = SequenceClassifier(seq_max_len, embed_w, tr.get_dict_size(), 2, n_layer, n_cell,
                           0.8, 1e-5, True)
    
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epoch):
        test_loss_mean_best = 100.0
        tr.reset_train_epoch()
        te.reset_test_epoch()
        train_acc_list=[]
        test_acc_list=[]
        train_loss_list=[]
        test_loss_list=[]
        n_tr_iter = int(tr.get_train_size() / batch_size)
        n_te_iter = int(te.get_test_size() / batch_size)
        for iteration in range(n_tr_iter):
            x, y, l, _ = tr.minibatch(batch_size)
            loss, acc = m.train_op(sess, x, y, l)
            train_acc_list.append(acc)
            train_loss_list.append(loss)
            print("Epoch = %d\t iter = %d/%d\tTrain Loss = %.3f\tAcc = %.3f"
                  % (epoch+1, iteration+1, n_tr_iter, loss, acc))
        for iteration in range(n_te_iter):
            x, y, l, _ = te.test_batch(batch_size)
            loss, acc = m.test_op(sess, x, y, l)
            test_acc_list.append(acc)
            test_loss_list.append(loss)
            print("Epoch = %d\t iter = %d/%d\tTest Loss = %.3f\tAcc = %.3f"
                  % (epoch+1, iteration+1, n_te_iter, loss, acc))
            
        test_loss_mean = numpy.mean(test_loss_list)
        train_loss_mean = numpy.mean(train_loss_list)
        test_acc_mean = numpy.mean(test_acc_list)
        train_acc_mean = numpy.mean(train_acc_list)
        if test_loss_mean < test_loss_mean_best:
            test_loss_mean_best = test_loss_mean
            m.save(sess, model_save_path)
        write_log('Epoch '+str(epoch+1)+'\ttrain loss = '+str(train_loss_mean)
                +'\ttest loss = '+str(test_loss_mean)+"\ttrain acc = "+str(train_acc_mean)
                +'\ttrain acc = '+str(test_acc_mean),
                log_save_path)