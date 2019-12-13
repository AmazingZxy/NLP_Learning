#!coding=utf8

"""
双向最大匹配算法
从文本的两边开始匹配
1、选取词数最少的为准
2、分词结果词数相同，没有歧义
不同就，返回单子字最少的那个

"""

class RMM(object):
    def __init__(self):
        self.windows = 3

    def cut(self, text):
        result = []
        index = len(text)
        dic = ['研究','研究生','生命','命','的','起源']
        # 控制整个文本是否匹配完成
        while index > 0:
            # 匹配成功直接跳过匹配的长度
            # 若没有匹配成功就从刚刚没匹配成功的字符串取[1:]在走匹配
            # index 控制每一次的文本的未被匹配的最后的位置
            for size in range(index-self.windows, index, 1):
                piece = text[size:index]
                if piece in dic:
                    index = size + 1
                    break

            index = index - 1
            result.append(piece)
        result.reverse()
        return result

class MM(object):
    def __init__(self):
        self.windows = 3

    def cut(self, text):
        result = []
        index = 0
        text_length = len(text)
        dic = ['研究','研究生','生命','命','的','起源']
        # 控制真个文本是否匹配完成
        while text_length > index:
            # 匹配成功直接跳过匹配的长度
            # 若没有匹配成功就从刚刚没匹配成功的字符串取[:-1]在走匹配
            for size in range(self.windows + index, index, -1):
                piece = text[index:size]
                if piece in dic:
                    index = size - 1
                    break

            index = index +1
            result.append(piece)
        return  result


if __name__ == '__main__':
    text = "研究生命的起源"
    rnn_tokenizer = RMM()
    nn_tokenizer = MM()
    b = rnn_tokenizer.cut(text)
    f = nn_tokenizer.cut(text)
    if len(f) < len(b): # 词数最少优先级最高
        print(f)
    elif len(f) >= len(b):
        print(b)
    else:
        if sum(map(lambda x: len(x) == 1, f)) > sum(map(lambda x: len(x) == 1, b)):
            print(b)
        else:
            print(f)

