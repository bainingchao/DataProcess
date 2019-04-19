# coding:utf8

"""
Description:LSA/LSI 潜在语义分析/索引
Author：伏草惟存
Prompt: code in Python3 env
"""

from mydict import *
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle as pkl
# python的pickle模块实现了基本的数据序列和反序列化。
# 通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储。
# 通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。

'''
作者：黄小猿
主题模型（LDA）(一)--通俗理解与简单应用
https://blog.csdn.net/qq_39422642/article/details/78730662

什么是LDA？
它是一种无监督的贝叶斯模型。
是一种主题模型，它可以将文档集中的每篇文档按照概率分布的形式给出。
是一种无监督学习，在训练时不需要手工标注的训练集，需要的是文档集和指定主题的个数。
是一种典型的词袋模型，它认为一篇文档是由一组词组成的集合，词与词之间没有顺序和先后关系。
'''

# LSA 潜在语义分析
def gensim_Corpus(corpus=None):
    dictionary = corpora.Dictionary(corpus)
    # 1 doc_bow转化成tfidf向量
    doc_bow_corpus = [dictionary.doc2bow(doc_cut) for doc_cut in corpus]
    tfidf_model = models.TfidfModel(dictionary=dictionary)  # 生成tfidf模型
    tfidf_corpus = [tfidf_model[doc_bow] for doc_bow in doc_bow_corpus]  # 将每doc_bow转换成对应的tfidf_doc向量
    print('doc_bow转换成对应的tfidf_doc向量:\n',tfidf_corpus)

    # 2 生成lsi model
    lsi_model = models.LsiModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=10)
    # 转换成lsi向量
    lsi_corpus = [lsi_model[tfidf_doc] for tfidf_doc in tfidf_corpus]
    print('LSA生成主题:\n',lsi_corpus)

    # 3 将lsi模型存储到磁盘上
    savepath =r'../dataSet/files/lsi_model.pkl'
    lsi_file = open(savepath, 'wb')
    pkl.dump(lsi_model, lsi_file)
    lsi_file.close()
    print('--- lsi模型已经生成 ---')


if __name__=='__main__':
    # corpus参数样例数据如下：
    corpus,classVec = loadDataSet()
    gensim_Corpus(corpus)
