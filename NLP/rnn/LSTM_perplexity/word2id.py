# _*_ encoding=utf8 _*_

"""
处理好每个单词的统计数据后，再将训练文件没测试文件根据单词表id化
"""

import codecs
import config



with codecs.open(config.VOCAB_OPTPUT, 'r', 'utf-8') as fr:
    vocab = {word.strip() for word in fr.readlines()}

word2id = {k:v for (k,v) in zip(vocab,range(len(vocab)))}

# 低频次用unk替换
def get_id(word):
    return word2id[word] if word in word2id else word2id['<unk>']

sort_word2id = sorted(word2id.items(),key = lambda x:x[1],reverse = True)
with codecs.open(config.word_dict, "w", encoding="utf-8") as fw:
    for (key,value) in sort_word2id:
        fw.write(key + "\t" + str(value) + "\n")

with codecs.open(config.train2id_data, "w", encoding="utf-8") as fw:
    with codecs.open(config.train_data, "r", encoding="utf-8") as fr:
        for index,line in enumerate(fr):
            words = line.strip().split() + ["<eos>"]
            out_line = ' '.join([str(get_id(word)) for word in words])
            fw.write(out_line + "\n")

print("词典生成完成，训练文件id化完成")

with codecs.open(config.test2id_data, "w", encoding="utf-8") as fw:
    with codecs.open(config.test_data, "r", encoding="utf-8") as fr:
        for index,line in enumerate(fr):
            words = line.strip().split() + ["<eos>"]
            out_line = ' '.join([str(get_id(word)) for word in words])
            fw.write(out_line + "\n")

print("词典生成完成，测试文件id化完成")

with codecs.open(config.valid2id_data, "w", encoding="utf-8") as fw:
    with codecs.open(config.valid_data, "r", encoding="utf-8") as fr:
        for index,line in enumerate(fr):
            words = line.strip().split() + ["<eos>"]
            out_line = ' '.join([str(get_id(word)) for word in words])
            fw.write(out_line + "\n")

print("词典生成完成，验证文件id化完成")
