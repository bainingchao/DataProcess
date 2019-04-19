# coding:utf8

"""
Description:生成lsi特征（潜在语义索引）
Author：伏草惟存
Prompt: code in Python3 env
"""

from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from mydict import *
import pickle as pkl


# 生成lsi特征（潜在语义索引）
def gensim_Corpus(corpus=None):
    dictionary = corpora.Dictionary(corpus)
    # 转换成doc_bow
    doc_bow_corpus = [dictionary.doc2bow(doc_cut) for doc_cut in corpus]
    # print(doc_bow_corpus)
    # 生成tfidf特征
    tfidf_model = models.TfidfModel(dictionary=dictionary)  # 生成tfidf模型
    tfidf_corpus = [tfidf_model[doc_bow] for doc_bow in doc_bow_corpus]  # 将每doc_bow转换成对应的tfidf_doc向量
    print('TFIDF:\n',tfidf_corpus)

    lsi_model = models.LsiModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=10)  # 生成lsi model
    # 生成corpus of lsi
    lsi_corpus = [lsi_model[tfidf_doc] for tfidf_doc in tfidf_corpus]  # 转换成lsi向量
    print('LSI:\n',lsi_corpus)
    # 将lsi模型存储到磁盘上
    savepath =r'../dataSet/files/lsi_model.pkl'
    lsi_file = open(savepath, 'wb')
    pkl.dump(lsi_model, lsi_file)
    lsi_file.close()
    print('--- lsi模型已经生成 ---')

if __name__=='__main__':
    # corpus参数样例数据如下：
    corpus,classVec = loadDataSet()
    gensim_Corpus(corpus)
