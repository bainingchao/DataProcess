# coding:utf-8

"""
Description: 6万条新闻文本数据清洗
Prompt: code in Python3 env
"""

import os,re,time
from REdealText import textParse
#******************** 高效读取文件***********************************


class loadFolders(object):   # 迭代器
    def __init__(self, par_path):
        self.par_path = par_path
    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath): # if file is a folder
                yield file_abspath

class loadFiles(object):
    def __init__(self, par_path):
        self.par_path = par_path
    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:              # level directory
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):     # secondary directory
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    this_file = open(file_path, 'rb') #rb读取方式更快
                    content = this_file.read().decode('utf8')
                    yield catg, content
                    this_file.close()


# 正则对字符串清洗
def textParse(str_doc):
    # 正则过滤掉特殊符号、标点、英文、数字等。
    r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:：;；|<=>?@，—。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    # 去除空格
    r2 = '\s+'
    # 去除换行符
    str_doc=re.sub(r1, ' ', str_doc)
    # 多个空格成1个
    str_doc=re.sub(r2, ' ', str_doc)
    # 去除换行符
    # str_doc = str_doc.replace('\n',' ')
    return str_doc


if __name__=='__main__':
    start = time.time()

    filepath = os.path.abspath(r'../dataSet/CSCMNews6w')
    files = loadFiles(filepath)
    n = 2  # n 表示抽样率， n抽1
    for i, msg in enumerate(files):
        if i % n == 0:
            catg = msg[0]
            file = msg[1]
            file = textParse(file)
            if int(i/n) % 1000 == 0:
                print('{t} *** {i} \t docs has been dealed'
                      .format(i=i, t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())),'\n',catg,':\t',file[:20])

    end = time.time()
    print('total spent times:%.2f' % (end-start)+ ' s')






