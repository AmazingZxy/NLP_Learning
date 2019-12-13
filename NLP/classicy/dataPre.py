# encoding:utf-8

import numpy as np
import os
from os.path import isfile, join

wordsList = np.load('wordsList.npy')
print('载入word列表')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8')
             for word in wordsList]
wordVectors = np.load('wordVectors.npy')
print('载入文本向量')

print(len(wordsList))
print(wordVectors.shape)



pos_files = ['pos/' + f for f in os.listdir(
    'pos/') if isfile(join('pos/', f))]
neg_files = ['neg/' + f for f in os.listdir(
    'neg/') if isfile(join('neg/', f))]
num_words = []
for pf in pos_files:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('正面评价完结')

for nf in neg_files:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('负面评价完结')

num_files = len(num_words)
print('文件总数', num_files)
print('所有的词的数量', sum(num_words))
print('平均文件词的长度', sum(num_words) / len(num_words))

if __name__ == '__main__':
    pass