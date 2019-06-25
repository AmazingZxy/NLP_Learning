# _*_ encoding=utf8 _*_

import numpy as np
import tensorflow as tf
import os
from get_batch import read_data,make_batches
import config


tf.app.flags.DEFINE_string('TRAIN_DATA', config.train2id_data ,"训练数据")
tf.app.flags.DEFINE_string('EVAL_DATA', config.test2id_data ,"测试数据")
tf.app.flags.DEFINE_string('TEST_DATA', config.valid2id_data ,"验证数据")

tf.app.flags.DEFINE_integer('HIDDEN_SIZE', 300 ,"影藏层数量")
tf.app.flags.DEFINE_integer('NUM_LAYERS', 2 ,"LSTM层数")
tf.app.flags.DEFINE_integer('VOCAB_SZIE', 10000 ,"词典大小")
tf.app.flags.DEFINE_integer('TRAIN_BATCH_SIZE', 20, 'batch_size的大小')
tf.app.flags.DEFINE_integer('TRAIN_NUM_STEP', 35, '训练数据的截断长度')
tf.app.flags.DEFINE_integer('EVAL_BATCH_SIZE', 1, '测试数据的batch大小')
tf.app.flags.DEFINE_integer('EVAL_NUM_STEP', 1, '测试数据的截断长度')
tf.app.flags.DEFINE_integer('NUM_EPOCH', 5, '训练的轮数')
tf.app.flags.DEFINE_float('LSTM_KEEP_PROB', 0.9, 'LSTM不被dropout的概率')
tf.app.flags.DEFINE_float('EMBEDDING_KEEP_PROB', 0.9, '词向量不被dropout的概率')
tf.app.flags.DEFINE_integer('MAX_GRAD_NORM', 5 ,"用户控制梯度膨胀的梯度大小的上限")
tf.app.flags.DEFINE_boolean('SHARE_EMB_AND_SOFTMAX', True ,"在softmax和词向量层共享参数")


FLAGS = tf.app.flags.FLAGS