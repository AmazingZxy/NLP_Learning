#!coding=utf8

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
            result.append(text[index] + '----') # 扫描单个字符
            # 匹配成功直接跳过匹配的长度
            # 若没有匹配成功就从刚刚没匹配成功的字符串取[:-1]在走匹配
            for size in range(self.windows + index, index, -1):  # 所有可能的结尾
                piece = text[index:size]  # 当前位置到结尾的连续字符串
                if piece in dic: # 判断是否在词典中
                    index = size - 1 # 移动位置
                    break

            index = index +1
            result.append(piece + '----')
        print(result)


if __name__ == '__main__':
    text = "研究生命的起源"
    tokenizer = MM()
    tokenizer.cut(text)

