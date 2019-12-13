#!coding=utf8

# 第一步：特征提取

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    """
    CountVectorizer会将文本中的词语转换为词频矩阵，
    它通过fit_transform函数计算各个词语出现的次数。

    :param corpus:
    :param ngram_range:
    :return:
    """
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features





def tfidf_transformer(bow_matrix):
    """
    TfidfTransformer用于统计vectorizer中每个词语的TFIDF值
    :param bow_matrix:
    :return:
    """
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix





def tfidf_extractor(corpus, ngram_range=(1, 1)):
    """
    即TfidfVectorizer类将CountVectorizer和TfidfTransformer类封装在一起。
    :param corpus:
    :param ngram_range:
    :return:
    """
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features





