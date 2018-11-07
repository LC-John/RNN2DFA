#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:42:24 2018

@author: zhanghuangzhao
"""

import tensorflow as tf
import numpy
import os

import config
import dataset
import classifier

from util import write_log

flags = tf.app.flags.FLAGS

def main(_):
    
    is_training = flags.is_training
    tomita_idx = flags.tomita
    cell_type = flags.cell
    seq_max_len = flags.max_seq_len
    embed_w = flags.embed_w
    n_cell = flags.cell_n
    n_epoch = flags.epoch_n
    batch_size = flags.batch_size
    dataset_path = os.path.join(flags.dataset_root,
                                "tomita_"+str(tomita_idx)+".pkl")
    log_save_path = os.path.join(flags.log_root,
                                 "tomita_"+str(tomita_idx)+"_"+cell_type+".log")
    model_root = os.path.join(flags.model_root,
                              "tomita_"+str(tomita_idx)+"_"+cell_type+"/")
    if not os.path.exists(model_root):
        os.system("mkdir "+model_root)
    model_save_path = os.path.join(model_root, "model.ckpt")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu
    gpus = flags.gpu.split(",")
    n_gpu = len(gpus)
    model = classifier.SequenceClassifier(seq_max_len, embed_w, 3, 2, n_cell,
                                          cell_type, 0.8, 1e-5, n_gpu, True)
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    sess.run(tf.global_variables_initializer())
    
    if is_training: # train
        os.system("rm -rf "+log_save_path)
        data = dataset.Dataset(dataset_path, seq_max_len)
        for epoch in range(n_epoch):
            test_loss_mean_best = 100.0
            data.reset_train_epoch()
            data.reset_test_epoch()
            train_acc_list=[]
            test_acc_list=[]
            train_loss_list=[]
            test_loss_list=[]
            n_tr_iter = int(data.get_train_size() / batch_size / n_gpu)
            n_te_iter = int(data.get_test_size() / batch_size / n_gpu)
            for iteration in range(n_tr_iter):
                x, y, l = data.minibatch(batch_size * n_gpu)
                loss, acc = model.train_op(sess, x, y, l)
                train_acc_list.append(acc)
                train_loss_list.append(loss)
                if (iteration % 100 == 0):
                    print("Epoch = %d\t iter = %d/%d\tTrain Loss = %.3f\tAcc = %.3f"
                          % (epoch+1, iteration+1, n_tr_iter, loss, acc))
            for iteration in range(n_te_iter):
                x, y, l = data.test_batch(batch_size * n_gpu)
                loss, acc = model.test_op(sess, x, y, l)
                test_acc_list.append(acc)
                test_loss_list.append(loss)
                if (iteration % 100 == 0):
                    print("Epoch = %d\t iter = %d/%d\tTest Loss = %.3f\tAcc = %.3f"
                          % (epoch+1, iteration+1, n_te_iter, loss, acc))
                
            test_loss_mean = numpy.mean(test_loss_list)
            train_loss_mean = numpy.mean(train_loss_list)
            test_acc_mean = numpy.mean(test_acc_list)
            train_acc_mean = numpy.mean(train_acc_list)
            if test_loss_mean < test_loss_mean_best:
                test_loss_mean_best = test_loss_mean
                model.save(sess, model_save_path)
            write_log('Epoch '+str(epoch+1)+'\ttrain loss = '+str(train_loss_mean)
                    +'\ttest loss = '+str(test_loss_mean)+"\ttrain acc = "+str(train_acc_mean)
                    +'\ttest acc = '+str(test_acc_mean)+'\n',
                    log_save_path)
    else:
        pass

if __name__ == "__main__":
    
    tf.app.run(main=main)