import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import random
from matplotlib.font_manager import FontProperties

from collections import Counter

from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import ward, dendrogram

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from normalization import normalize_corpus
from sklearn.cluster import KMeans



def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    """
    构建tf-idf特征矩阵
    :param documents: 用空格连接的query
    :param feature_type: 不同的特征情况，词袋模型。tfidf模型 还有frequency
    :param ngram_range:
    :param min_df:
    :param max_df:
    :return:
    """
    feature_type = feature_type.lower().strip()

    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix

# 查看豆瓣读书的文件的信息
# python3 str的默认类型是unicode编码。
book_data = pd.read_csv('data/data.csv').astype(str) #读取文件
print(book_data.head())

# 获取title以及content这两列的数据
book_titles = book_data['title'].tolist()
book_content = book_data['content'].tolist()

print('书名:', book_titles[0])
print('内容:', book_content[0][:10])


# normalize corpus
# 获取每一列content中的内容的分词信息
# eg 在 第一次世界大战 的 硝烟 中
norm_book_content = normalize_corpus(book_content)


# 提取 tf-idf 特征
# 可以知道每个词的tfidf值
vectorizer, feature_matrix = build_feature_matrix(norm_book_content,
                                                  feature_type='tfidf',
                                                  min_df=0.2, max_df=0.90,
                                                  ngram_range=(1, 2))
# 查看特征的形状
print("feature_matrix.shape:")
# shape[0] 是问句数，shape[1]是所有词的数量
print(feature_matrix.shape)

# 获取特征名字
# get_feature_names函数可以获取词典中所有的词，这儿展示了前10个
feature_names = vectorizer.get_feature_names()

# 打印某些特征
print(feature_names[:10])




def k_means(feature_matrix, num_clusters=10):
    """
    提取特征后开始聚类
    :param feature_matrix:
    :param num_clusters:
    :return:
    """
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


num_clusters = 10
km_obj, clusters = k_means(feature_matrix=feature_matrix,
                           num_clusters=num_clusters)
book_data['Cluster'] = clusters



# 获取每个cluster的数量
c = Counter(clusters)
print(c.items())


def get_cluster_data(clustering_obj, book_data,
                     feature_names, num_clusters,
                     topn_features=10):
    """
    得到每一个聚类的数据
    :param clustering_obj:
    :param book_data:
    :param feature_names:
    :param num_clusters:
    :param topn_features:
    :return:
    """
    cluster_details = {}
    # 获取cluster的center
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # 获取每个cluster的关键特征
    # 获取每个cluster的书
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index]
                        for index
                        in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        books = book_data[book_data['Cluster'] == cluster_num]['title'].values.tolist()
        cluster_details[cluster_num]['books'] = books

    return cluster_details


def print_cluster_data(cluster_data):
    """
    打印每一个聚类的书的名字
    :param cluster_data:
    :return:
    """
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-' * 20)
        print('Key features:', cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)





def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, book_data,
                  plot_size=(16, 8)):
    # generate random color for clusters
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color

    # define markers for clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=1)
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data[0:500].items():
        # assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # map each unique cluster label with its coordinates and books
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': book_data['Cluster'].values.tolist(),
                                       'title': book_data['title'].values.tolist()
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # plot each cluster using co-ordinates and book titles
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off',
                       labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=fontP)
    # add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'],
                cluster_plot_frame.ix[index]['y'],
                cluster_plot_frame.ix[index]['title'], size=8)
        # show the plot
    plt.show()


cluster_data = get_cluster_data(clustering_obj=km_obj,
                                book_data=book_data,
                                feature_names=feature_names,
                                num_clusters=num_clusters,
                                topn_features=5)

print_cluster_data(cluster_data)

plot_clusters(num_clusters=num_clusters,
              feature_matrix=feature_matrix,
              cluster_data=cluster_data,
              book_data=book_data,
              plot_size=(16, 8))




def affinity_propagation(feature_matrix):
    sim = feature_matrix * feature_matrix.T
    sim = sim.todense()
    ap = AffinityPropagation()
    ap.fit(sim)
    clusters = ap.labels_
    return ap, clusters


# get clusters using affinity propagation
ap_obj, clusters = affinity_propagation(feature_matrix=feature_matrix)
book_data['Cluster'] = clusters

# get the total number of books per cluster
c = Counter(clusters)
print(c.items())

# get total clusters
total_clusters = len(c)
print('Total Clusters:', total_clusters)

cluster_data = get_cluster_data(clustering_obj=ap_obj,
                                book_data=book_data,
                                feature_names=feature_names,
                                num_clusters=total_clusters,
                                topn_features=5)

print_cluster_data(cluster_data)

plot_clusters(num_clusters=num_clusters,
              feature_matrix=feature_matrix,
              cluster_data=cluster_data,
              book_data=book_data,
              plot_size=(16, 8))


def ward_hierarchical_clustering(feature_matrix):
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix


def plot_hierarchical_clusters(linkage_matrix, book_data, figure_size=(8, 12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size)
    book_titles = book_data['title'].values.tolist()
    # plot dendrogram
    ax = dendrogram(linkage_matrix, orientation="left", labels=book_titles)
    plt.tick_params(axis='x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('ward_hierachical_clusters.png', dpi=200)


# build ward's linkage matrix
linkage_matrix = ward_hierarchical_clustering(feature_matrix)
# plot the dendrogram
plot_hierarchical_clusters(linkage_matrix=linkage_matrix,
                           book_data=book_data,
                           figure_size=(8, 10))
