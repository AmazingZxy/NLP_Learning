# _*_ encoding=utf8 _*_

import numpy as np
import codecs
import config

train_batch_size = config.train_batch_size
train_step_num = config.train_step_num


def read_data(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as fr:
        id_string = ' '.join([line.strip() for line in fr.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batches(id_list, batch_size, num_step):
    # 计算总的batch_size数，每一个batch包含的单词数量是batch_size*num_step
    num_batch = (len(id_list) - 1) // (batch_size * num_step)

    # 把数据整理成【batch_size, num_batches*num_step】
    data = np.array(id_list[:num_batch * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batch * num_step])
    # 沿第二个维度将数据分成num_batch个batch,存入一个数组
    data_batches = np.split(data, num_batch, axis=1)

    # 每个位置右移一位，再构造RNN输出
    label = np.array(id_list[1:num_batch * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batch * num_step])
    label_batch = np.split(label, num_batch, axis=1)

    # print(data_batches[0])
    # 返回RNN的输入和输出
    return list(zip(data_batches, label_batch))

# make_batches(read_data(config.train2id_data),train_batch_size,train_step_num)

if __name__ == '__main__':
    data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,21,22,23,24,25,26,27,28,29,30,31,32])
    data1= np.reshape(data,[2,3*4])  #2 3 4
    data2 = np.split(data1,3,axis=1) # 3 2 4
    print(data)
    print(data1)
    print(data2)
    a = [1,2,3,4,5,6]
    print(data2[:,2,:])