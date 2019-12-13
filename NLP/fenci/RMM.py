#!coding=utf8

"""
逆向最大匹配算法
从文本的末端开始匹配
和正向最大匹配算法相反
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
            result.append(piece + '----')
        result.reverse()
        print(result)


if __name__ == '__main__':
    text = "研究生命的起源"
    tokenizer = RMM()
    tokenizer.cut(text)