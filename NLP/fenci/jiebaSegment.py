#!coding=utf8

import jieba

sent = "中文分词是文本处理不可缺少的一步"
seg_list = jieba.cut(sent, cut_all=True)
print("全模式")
print("/ ".join(seg_list))
print("==")

seg_list = jieba.cut(sent, cut_all = False)
print("精确模式")
print("/ ".join(seg_list))
print("==")

seg_list = jieba.cut(sent)
print("默认精确模式")
print("/ ".join(seg_list))
print("==")

seg_list = jieba.cut_for_search(sent)
print("搜索引擎模式")
print("/ ".join(seg_list))
print("==")



