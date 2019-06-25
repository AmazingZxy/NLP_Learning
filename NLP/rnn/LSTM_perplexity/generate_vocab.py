# _*_ encoding=utf8 _*_

import codecs
import collections
from operator import itemgetter
import config

counter = collections.Counter()
with codecs.open(config.train_data, 'r', encoding='utf-8') as fr:
    for line in fr:
        line = line.strip().split()
        for word in line:
            counter[word] += 1

sorted_word_to_counter = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_counter]
sorted_words.insert(0, "<eos>")

with codecs.open(config.VOCAB_OPTPUT, 'w', encoding='utf-8') as fw:
    for word in sorted_words:
        fw.write(word + "\n")

print("词汇表生成完成")