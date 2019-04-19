
# coding:utf8

"""
Description:gensim计算tfidf
Author：伏草惟存
Prompt: code in Python3 env
"""

import os
from mydict import *
from gensim import corpora, models


# tfidf特征
def gensim_Corpus(corpus=None,classVec=''):
    dictionary = corpora.Dictionary(corpus)

    # 转换成doc_bow
    doc_bow_corpus = [dictionary.doc2bow(doc_cut) for doc_cut in corpus]
    print('词袋模型实现文本向量化：\n',doc_bow_corpus)

    # 生成tfidf特征
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    print('生成tfidf模型 ：\n',tfidf_model) #num_nnz是每个文件中不重复词个数的和

    corpus_tfidf = {} # tfidf字典
    i=0 # 获取类别
    for doc_bow in doc_bow_corpus:
        file_tfidf = tfidf_model[doc_bow] # 词袋填充
        catg = classVec[i]                # 类别
        tmp = corpus_tfidf.get(catg, [])
        tmp.append(file_tfidf)
        print('第',i,'类下特征词的TFIDF：\n',tmp)
        if tmp.__len__() == 1: # 某篇文章成功
            corpus_tfidf[catg] = tmp
        i+=1
    # 本地化存储
    print('tfidf字典:\n',corpus_tfidf)
    catgs = list(corpus_tfidf.keys())
    for catg in catgs:
        savepath =r'../dataSet/files/tfidf_corpus'
        corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=savepath, s=os.sep, c=catg),corpus_tfidf.get(catg),id2word=dictionary)



if __name__=='__main__':
    # corpus参数样例数据如下：
    corpus,classVec = loadDataSet()
    gensim_Corpus(corpus,classVec)
