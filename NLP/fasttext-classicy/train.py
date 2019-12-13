# !/usr/bin/python
# -*- coding: UTF-8 -*-


import fasttext


model = fasttext.train_supervised(input="./data/train.txt")
model.save_model("./data/model_class.bin")
data = model.predict("这 特么 的 都是 垃圾 123")
print(data[0][0])
data = model.test("./data/test.txt")
print(data[0])
print(data[1])
print(data[2])