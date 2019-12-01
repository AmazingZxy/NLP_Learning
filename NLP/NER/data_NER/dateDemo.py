# -*- coding:utf-8 -*-

"""
时间识别的场景
基于语音问答的酒店预订系统中，识别时间
也就是把一些不标准的时间给找出来，识别成标准的时间。
"""
import re
from datetime import datetime,timedelta
from dateutil.parser import parse
import jieba
jieba.load_userdict("./data/dict.txt")

import jieba.posseg as psg



UTIL_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}

UTIL_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}

def cn2dig(src):
    if src == "":
        return None
    m = re.match("\d+", src)
    if m:
        return int(m.group(0))
    rsl = 0
    unit = 1
    for item in src[::-1]:
        if item in UTIL_CN_UNIT.keys():
            unit = UTIL_CN_UNIT[item]
        elif item in UTIL_CN_NUM.keys():
            num = UTIL_CN_NUM[item]
            rsl += num * unit
        else:
            return None
    if rsl < unit:
        rsl += unit
    return rsl

def year2dig(year):
    res = ''
    for item in year:
        if item in UTIL_CN_NUM.keys():
            res = res + str(UTIL_CN_NUM[item])
        else:
            res = res + item
    m = re.match("\d+", res)
    if m:
        if len(m.group(0)) == 2:
            return int(datetime.datetime.today().year/100)*100 + int(m.group(0))
        else:
            return int(m.group(0))
    else:
        return None

def parse_datetime(msg):
    """
    时间解析
    :param msg:
    :return:
    """
    if msg is None or len(msg) == 0:
        return None
    # 正则分维度，方便接下来填槽处理值
    m = re.match(
        r"([0-9零一二两三四五六七八九十]+年)?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?([上中下午晚早]+)?([0-9零一二两三四五六七八九十百]+[点:\.时])?([0-9零一二三四五六七八九十百]+分?)?([0-9零一二三四五六七八九十百]+秒)?",
        msg)

    if m.group(0) is not None:
        # 分维度
        res = {
            "year": m.group(1),
            "month": m.group(2),
            "day": m.group(3),
            "hour": m.group(5) if m.group(5) is not None else '00',
            "minute": m.group(6) if m.group(6) is not None else '00',
            "second": m.group(7) if m.group(7) is not None else '00',
        }
        params = {}

        for name in res:
            if res[name] is not None and len(res[name]) != 0:
                tmp = None
                if name == 'year':
                    tmp = year2dig(res[name][:-1])
                else:
                    tmp = cn2dig(res[name][:-1])
                if tmp is not None:
                    params[name] = int(tmp)

        target_date = datetime.today().replace(**params)

        # 区分早中晚,是早上还是下午
        is_pm = m.group(4)
        if is_pm is not None:
            if is_pm == u'下午' or is_pm == u'晚上' or is_pm =='中午':
                hour = target_date.time().hour
                if hour < 12:
                    target_date = target_date.replace(hour=hour + 12)
        return target_date.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return None



# def check_time_valid(word):
#     m = re.match("\d+$", word)
#     if m:
#         if len(word) <= 6:
#             return None
#     word1 = re.sub('[号|日]\d+$', '日', word)
#     if word1 != word:
#         return check_time_valid(word1)
#     else:
#         return word1


def time_extract(text):
    """
    时间提取函数
    :param text: 用户query
    :return:
    """
    # 搜集时间词
    time_res = []
    word = ''
    # 关键日期提取，保证这些词的信息一定要提取出来
    keyDate = {'今天': 0, '明天':1, '后天': 2, '前天': 3, '昨天': 4}
    for k, v in psg.cut(text):  # 结巴分词 + 词性获取
        if k in keyDate: # keyDate都属于时间词，
            if word != '':
                time_res.append(word)
            # 默认日是今天
            word = (datetime.today() + timedelta(days=keyDate.get(k, 0))).strftime('%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日')
        elif word != '':
            if v in ['m', 't']: # 数字和时间词性
                word = word + k
            else:
                time_res.append(word)
                word = ''
        elif v in ['m', 't']:
            word = k

    if word != '':
        time_res.append(word)
    # result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))
    result = time_res
    final_res = [parse_datetime(w) for w in result]
    return [x for x in final_res if x is not None]


if __name__ == '__main__':
    text1 = '我要住到明天下午三点'
    print(text1, time_extract(text1), sep=':')

    text2 = '预定28号的房间'
    print(text2, time_extract(text2), sep=':')

    text3 = '我要从26号下午4点住到11月2号'
    print(text3, time_extract(text3), sep=':')

    text4 = '我要预订今天到30的房间'
    # 这个识别不了是因为，这个30并不是很明确，强行加规则肯定不太好， 30号就能识别的
    print(text4, time_extract(text4), sep=':')

    text5 = '今天30号呵呵'
    print(text5, time_extract(text5), sep=':')