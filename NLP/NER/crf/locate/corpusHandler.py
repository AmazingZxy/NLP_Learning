#!coding=utf-8

import codecs

"""
数据处理

"""
org_query_file = './data/people-daily.txt'

def tag_line(words, mark):
    """
    打标签，也就是对每一行进行打标签，给每一个词打上BIOS其中一个标签，
    :param words: 输入是一个query
    :param mark:
    :return: 输出是一个word  tag的键值对
    """
    chars = []
    tags = []
    temp_word = '' #用于合并组合词
    for word in words: # 遍历query每一个字符
        word = word.strip('\t ') # 去空格
        if temp_word == '':
            bracket_pos = word.find('[')
            w, h = word.split('/')
            if bracket_pos == -1:
                if len(w) == 0: continue
                chars.extend(w)
                if h == 'ns': # ns表示地名
                    tags += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                else:
                    tags += ['O'] * len(w)
            else:
                w = w[bracket_pos+1:]
                temp_word += w
        else:
            bracket_pos = word.find(']')
            w, h = word.split('/')
            if bracket_pos == -1:
                temp_word += w
            else:
                w = temp_word + w
                h = word[bracket_pos+1:]
                temp_word = ''
                if len(w) == 0: continue
                chars.extend(w)
                if h == 'ns':
                    tags += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                else:
                    tags += ['O'] * len(w)

    assert temp_word == ''
    return (chars, tags)

def corpusHandler(corpusPath):
    import os
    root = os.path.dirname(corpusPath)
    with codecs.open(corpusPath,"r","utf-8") as corpus_f, \
            codecs.open(os.path.join(root, 'train.txt'), 'w', "utf-8") as train_f, \
            codecs.open(os.path.join(root, 'test.txt'), 'w', "utf-8") as test_f:

        pos = 0
        for line in  corpus_f:
            line = line.strip('\r\n\t ') # 去除首尾空格
            if line == '': continue # 去除空行
            isTest = True if pos % 5 == 0 else False  # 抽样20%作为测试集使用

            #例子 19980101-01-001-003/m （/w 一九九七年/t 十二月/t 三十一日/t ）/w
            words = line.split()[1:] # 去掉前面的年月日
            if len(words) == 0: continue
            line_chars, line_tags = tag_line(words, pos)
            saveObj = test_f if isTest else train_f
            for k, v in enumerate(line_chars):
                saveObj.write(v + '\t' + line_tags[k] + '\n')
            saveObj.write('\n')
            pos += 1

if __name__ == '__main__':
    corpusHandler(org_query_file)