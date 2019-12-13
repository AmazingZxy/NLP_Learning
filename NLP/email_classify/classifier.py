#!coding=utf8

import codecs
import numpy as np
from sklearn.cross_validation import train_test_split

from normalization import normalize_corpus
from sklearn import metrics

from feature_extractors import bow_extractor, tfidf_extractor
import gensim
import jieba

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

def get_data():
    '''
    获取数据
    正常的邮件：ham_data.txt
    垃圾邮件：spam_data.txt
    :return: 文本数据，对应的labels
    '''
    with codecs.open("data/ham_data.txt", "r", encoding="utf8") as ham_f, codecs.open("data/spam_data.txt", "r", encoding="utf8") as spam_f:
        # 读取正常的邮件的内容,按列读取
        ham_data = ham_f.readlines()
        print(len(ham_data))
        # 读取错误的邮件的内容，按列读取
        spam_data = spam_f.readlines()
        print(len(spam_data))

        # 正向的每一条的标签都是1
        ham_label = np.ones(len(ham_data)).tolist()

        # 负面的每一条标签都是0
        spam_label = np.zeros(len(spam_data)).tolist()
        # 拼接所有文本组成一个list,每一个都是一列
        corpus = ham_data + spam_data

        # 对特征也是组成拼接成一个list，每一个元素是每一列数据对应的特征标签
        labels = ham_label + spam_label


    return corpus, labels


def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    '''
    将数据分为测试集和训练集
    :param corpus: 文本数据
    :param labels: label数据
    :param test_data_proportion:测试数据占比 
    :return: 训练数据,测试数据，训练label,测试label
    '''

    # 比例为0.3
    # train_test_split函数用于将矩阵随机划分为训练子集和测试子集，
    # 并返回划分好的训练集测试集样本和训练集测试集标签。
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,
                                                        test_size=test_data_proportion, random_state=42)
    return train_X, test_X, train_Y, test_Y


def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels





def get_metrics(true_labels, predicted_labels):
    print('准确率:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        2))
    print('精度:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2))
    print('召回率:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2))
    print('F1得分:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2))


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    """
    :param classifier:  分类器，贝叶斯、svm、lr
    :param train_features: 训练数据
    :param train_labels: 训练特征
    :param test_features:
    :param test_labels:
    :return:
    """
    # 构建分类模型
    classifier.fit(train_features, train_labels)
    # 使用模型进行预测
    predictions = classifier.predict(test_features)
    # 计算模型预测的结果
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions


def main():
    # 获取数据集
    corpus, labels = get_data()
    print("总的数据量:", len(labels))

    # 删除无用数据，也就是空数据
    corpus, labels = remove_empty_docs(corpus, labels)

    print('样本之一:', corpus[10])
    print('样本对应的label:', labels[10])
    label_name_map = ["垃圾邮件", "正常邮件"]
    # 下标为0 的是垃圾邮件，为1 的是正常邮件
    print('实际类型:', label_name_map[int(labels[10])], label_name_map[int(labels[5900])])

    # 对数据进行划分
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                            labels,
                                                                            test_data_proportion=0.3)


    # 进行归一化
    # 现在数据的个数
    # 第二个参数传入为True表示是否用分词信息
    norm_train_corpus = normalize_corpus(train_corpus)
    norm_test_corpus = normalize_corpus(test_corpus)
    print(norm_train_corpus[11])


    print("==========数据处理完成==========")



    # 词袋模型特征
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    # tfidf 特征
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    # jieba分词
    tokenized_train = [jieba.lcut(text)
                       for text in norm_train_corpus]
    print(tokenized_train[2:10])
    tokenized_test = [jieba.lcut(text)
                      for text in norm_test_corpus]
    # build word2vec 模型
    model = gensim.models.Word2Vec(tokenized_train,
                                   size=500,
                                   window=100,
                                   min_count=30,
                                   sample=1e-3)

    # 朴树贝叶斯分类器
    mnb = MultinomialNB()
    # svm分类器
    svm = SGDClassifier(loss='hinge', n_iter=100)
    # lr分类器
    lr = LogisticRegression()

    # 两种数据处理方式，三种分类模型

    # 基于词袋模型的多项朴素贝叶斯
    print("基于词袋模型特征的贝叶斯分类器")
    mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)

    # 基于词袋模型特征的逻辑回归
    print("基于词袋模型特征的逻辑回归")
    lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                      train_features=bow_train_features,
                                                      train_labels=train_labels,
                                                      test_features=bow_test_features,
                                                      test_labels=test_labels)

    # 基于词袋模型的支持向量机方法
    print("基于词袋模型的支持向量机")
    svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)


    # 基于tfidf的多项式朴素贝叶斯模型
    print("基于tfidf的贝叶斯模型")
    mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)
    # 基于tfidf的逻辑回归模型
    print("基于tfidf的逻辑回归模型")
    lr_tfidf_predictions=train_predict_evaluate_model(classifier=lr,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)


    # 基于tfidf的支持向量机模型
    print("基于tfidf的支持向量机模型")
    svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)



    import re

    num = 0
    # 用户问句，该问句的标签，该问句的预测结果
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        # 垃圾邮件
        if label == 0 and predicted_label == 0:
            print('邮件类型:', label_name_map[int(label)])
            print('预测的邮件类型:', label_name_map[int(predicted_label)])
            print('原始文本:')
            print(document)

            num += 1
            if num == 4:
                break

    num = 0

    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        if label == 1 and predicted_label == 0:
            print('邮件类型:', label_name_map[int(label)])
            print('预测的邮件类型:', label_name_map[int(predicted_label)])
            print('原始文本:')
            print(document)

            num += 1
            if num == 4:
                break


if __name__ == "__main__":
    main()
