#!coding=utf-8
import os
import codecs
import CRFPP

"""
这个文件的作用就是验证crf_test生成的测试出来的数据的准确率。

"""
def f1(path):
    """
    f1值，验证准确率
    精确率：记录正确的地理位置标记数/所有的地理位置的标签数
    召回率：记录正确的地理位置标记数/记录真实的地理位置标记数
    :param path:
    :return:
    """

    with codecs.open(path, "r", "utf-8") as f:

        all_tag = 0 #记录所有的标记数
        loc_tag = 0 #记录真实的地理位置标记数
        pred_loc_tag = 0 #记录预测的地理位置标记数
        correct_tag = 0 #记录正确的标记数
        correct_loc_tag = 0 #记录正确的地理位置标记数

        # 四种状态
        states = ['B', 'M', 'E', 'S']
        for line in f:
            line = line.strip()
            if line == '': continue
            _, r, p = line.split()# r 实际 p预测

            all_tag += 1

            if r == p:
                correct_tag += 1
                if r in states:
                    correct_loc_tag += 1
            if r in states: loc_tag += 1
            if p in states: pred_loc_tag += 1

        loc_P = 1.0 * correct_loc_tag/pred_loc_tag
        loc_R = 1.0 * correct_loc_tag/loc_tag
        print('loc_P:{0}, loc_R:{1}, loc_F1:{2}'.format(loc_P, loc_R, (2*loc_P*loc_R)/(loc_P+loc_R)))

def load_model(path):
    """
    -m 加载模型

    :param path:
    :return:
    """
    # -v 3: access deep information like alpha,beta,prob
    # -nN: enable nbest output. N should be >= 2
    if os.path.exists(path):
        return CRFPP.Tagger('-m {0} -v 3 -n2'.format(path))
    return None

def locationNER(text):
    """
    地点识别
    :param text:
    :return:
    """
    tagger = load_model('./model')

    for c in text:
        tagger.add(c)

    result = []
    tagger.parse()

    word = ''
    for i in range(0, tagger.size()):
        for j in range(0, tagger.xsize()):
            ch = tagger.x(i, j)
            tag = tagger.y2(i)
            if tag == 'B':
                word = ch
            elif tag == 'M':
                word += ch
            elif tag == 'E':
                word += ch
                result.append(word)
            elif tag == 'S':
                word = ch
                result.append(word)


    return result


if __name__ == '__main__':
    f1('./data/test.rst')

    text = '我中午要去北京饭店，下午去中山公园，晚上回亚运村。'
    print(text, locationNER(text), sep='==> ')

    text = '杭州的东边也就是上海的西边'
    print(text, locationNER(text), sep='==> ')

    text = '打的去北京南站地铁站'
    print(text, locationNER(text), sep='==> ')
