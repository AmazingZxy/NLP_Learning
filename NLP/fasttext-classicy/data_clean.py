# !/usr/bin/python
# -*- coding: UTF-8 -*-


import codecs
import jieba


train_file =  codecs.open("data/train.txt", "w", encoding="utf8")
test_file =  codecs.open("data/test.txt", "w", encoding="utf8")

# 正样本
with codecs.open("data/train_good.txt", "w", encoding="utf8") as fw:
    with codecs.open("data/ham_data.txt", "r", encoding="utf8") as f:
        for index,line in enumerate(f):
            line = line.rstrip()
            data = jieba.cut(line)
            data = " ".join(data)
            fw.write(data + "\t" + "__label__good" + "\n")
            if index % 10 == 1:
                test_file.write(data + "\t" + "__label__good" + "\n")
            else:
                train_file.write(data + "\t" + "__label__good" + "\n")

print("正样本数据准备完成")

# 负样本
with codecs.open("data/train_bad.txt", "w", encoding="utf8") as fw:
    with codecs.open("data/spam_data.txt", "r", encoding="utf8") as f:
        for index,line in enumerate(f):
            line = line.rstrip()
            data = jieba.cut(line)
            data = " ".join(data)
            fw.write(data + "\t" + "__label__bad" + "\n")

            if index % 10 == 1:
                test_file.write(data + "\t" + "__label__bad" + "\n")
            else:
                train_file.write(data + "\t" + "__label__bad" + "\n")

print("负样本数据准备完成")


train_file.close()
test_file.close()
