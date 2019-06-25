# _*_ encoding=utf8 _*_

import numpy as np
import tensorflow as tf
import os
from get_batch import read_data,make_batches
from model_config import FLAGS


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps # 截断长度
        self.input_data = tf.placeholder(tf.int32, [batch_size,num_steps])
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])

        dropout_keep_prob = FLAGS.LSTM_KEEP_PROB if is_training else 1.0

        # 定义单个基本的LSTM单元。同时指明影藏层的数量hidden_size
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.HIDDEN_SIZE)

        # 当state_is_tuple=True的时候，state是元组形式，state=(c,h)。如果是False，那么state是一个由c和h拼接起来的张量，
        # state=tf.concat(1,[c,h])。在运行时，返回2值，一个是h，还有一个state
        # rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.HIDDEN_SIZE，state_is_tuple=True)

        # dropout,就是指网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。这是一张正则化思想，可以有效防止过拟合
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob = dropout_keep_prob)

        lstm_cells = [
            rnn_cell for _ in range(FLAGS.NUM_LAYERS)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        # 将LSTM的状态初始化全为0 的数组,zero_state这个函数生成全零的初始状态，
        # initial_state包含了两个张量的LSTMStateTuple类，其中.c和.h分别是c状态和h状态
        # 每次使用一个batch大小的训练样本
        # 初始化的c 和 h的状态
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 定义单词的词向量矩阵 300维 [vocab_size, embedding_size]
        embedding = tf.get_variable("embedding", [FLAGS.VOCAB_SZIE, FLAGS.HIDDEN_SIZE])
        # 将单词转化为词向量
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, FLAGS.EMBEDDING_KEEP_PROB)

        outputs = []
        # 初始的c h状态
        state = self.initial_state
        # LSTM循环
        # 安装文本的顺序向cell输入
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                # output: shape[num_steps][batch,hidden_size]
                outputs.append(cell_output)

        # axis = 0 代表在第0个维度拼接
        # axis = 1  代表在第1个维度拼接
        # 把之前outputs展开，成[batch, hidden_size*num_steps],
        # 然后 reshape, 成[batch*numsteps, hidden_size]
        output = tf.reshape(tf.concat(outputs, 1), [-1, FLAGS.HIDDEN_SIZE])

        # 获取embedding的权重 [HIDDEN_SIZE, VOCAB_SZIE]
        if FLAGS.SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight", [FLAGS.HIDDEN_SIZE, FLAGS.VOCAB_SZIE])

        bias = tf.get_variable("bias", [FLAGS.VOCAB_SZIE])
        # [batch*numsteps,VOCAB_SZIE]
        logits = tf.matmul(output, weight) + bias

        loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits
        )
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training: return

        trainable_variables = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables),FLAGS.MAX_GRAD_NORM)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))






