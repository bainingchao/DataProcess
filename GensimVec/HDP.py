
# coding:utf8

"""
Description:HDP 分层狄利克雷过程
Author：伏草惟存
Prompt: code in Python3 env
"""

from mydict import *
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle as pkl


'''
狄利克雷分布通俗讲解
https://blog.csdn.net/yan456jie/article/details/52170772

其不同之处有两点：
1、HDP中引入了Dirichlet过程，而LDA中是Dirichlet分布。
2、在表示上，HDP中的分布本身是可以作为一个变量的。
'''

## 分层狄利克雷过程HDP
def gensim_Corpus(corpus=None):
    dictionary = corpora.Dictionary(corpus)
    # 1 doc_bow转化为tfidf向量
    doc_bow_corpus = [dictionary.doc2bow(doc_cut) for doc_cut in corpus]
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    tfidf_corpus = [tfidf_model[doc_bow] for doc_bow in doc_bow_corpus]
    print('doc_bow转换成对应的tfidf_doc向量:\n',tfidf_corpus)

    # 2 分层狄利克雷过程（Hierarchical Dirichlet Process，HDP ,一种无参数贝叶斯方法）
    hdp_model = models.HdpModel(doc_bow_corpus, id2word=dictionary)
    hdp_corpus = [hdp_model[doc_bow] for doc_bow in doc_bow_corpus]  # 转换成HDP向量
    print('HDP :\n',hdp_corpus)

    # 3 将RP模型存储到磁盘上
    savepath =r'../dataSet/files/hdp_model.pkl'
    hdp_file = open(savepath, 'wb')
    pkl.dump(hdp_model, hdp_file)
    hdp_file.close()
    print('--- HDP模型已经生成 ---')

if __name__=='__main__':
    # corpus参数样例数据如下：
    corpus,classVec = loadDataSet()
    gensim_Corpus(corpus)
