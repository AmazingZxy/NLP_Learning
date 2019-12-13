#!coding=utf8

def full_segment(text, wordDict):
    word_list = []
    for i in range(len(text)):
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in wordDict:
                word_list.append(word)
    return word_list


if __name__ == '__main__':
    list = ["商品","和服","服务","商","品","和","服","务"]
    str = "商品和服务"
    word_list = full_segment(str, list)
    print(word_list)