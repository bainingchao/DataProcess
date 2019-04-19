
# coding:utf8

"""
Description:gensim包进行特征词提取
Author：伏草惟存
Prompt: code in Python3 env
"""

from gensim import corpora


# 构建语料词典
def gensim_Corpus(corpus=None):
    # 1 词典
    dictionary = corpora.Dictionary(corpus)
    print('构造语料词典：\n',dictionary)

    # 2 删除停用词和仅出现一次的词
    # once_ids = [dictionary[tokenid] for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    print('仅出现一次的词：\n',once_ids)
    dictionary.filter_tokens(once_ids)
    print('自定义选择后的特征词典：\n',dictionary)
    # 消除id序列在删除词后产生的不连续的缺口
    dictionary.compactify()

    # 3 dict 方式存储词典
    savePath = r'../dataSet/files/mycorpus.dict'
    dictionary.save(savePath)  # 把字典保存起来，方便以后使用

    mydict = corpora.Dictionary.load(savePath)
    print('加载Dict词典：\n',mydict)

    # 4 Txt 方式存储词典
    savePath = r'../dataSet/files/mycorpus.txt'
    dictionary.save_as_text(savePath) # 保存文本

    list2 = corpora.Dictionary.load_from_text(savePath) # 加载文本
    print('加载txt词典：\n',list2)



'''创建数据集：单词列表postingList, 所属类别classVec'''
def loadDataSet():
    # corpus参数样例数据如下：
    corpus =[]
    tiyu = ['姚明', '我来', '承担', '连败', '巨人', '宣言', '酷似', '当年', '麦蒂', '新浪', '体育讯', '北京', '时间', '消息', '休斯敦', '纪事报', '专栏', '记者', '乔纳森', '费根', '报道', '姚明', '渴望', '一场', '胜利', '当年', '队友', '麦蒂', '惯用', '句式']
    yule = ['谢婷婷', '模特', '酬劳', '仅够', '生活', '风光', '背后', '惨遭', '拖薪', '新浪', '娱乐', '金融', '海啸', 'blog', '席卷', '全球', '模特儿', '酬劳', '被迫', '打折', '全职', 'Model', '谢婷婷', '业界', '工作量', '有增无减', '收入', '仅够', '糊口', '拖薪']
    jioayu = ['名师', '解读', '四六级', '阅读', '真题', '技巧', '考前', '复习', '重点', '历年', '真题', '阅读', '听力', '完形', '提升', '空间', '天中', '题为', '主导', '考过', '六级', '四级', '题为', '主导', '真题', '告诉', '方向', '会考', '题材', '包括']
    shizheng = ['美国', '军舰', '抵达', '越南', '联合', '军演', '中新社', '北京', '日电', '杨刚', '美国', '海军', '第七', '舰队', '三艘', '军舰', '抵达', '越南', '岘港', '为期', '七天', '美越', '南海', '联合', '军事训练', '拉开序幕', '美国', '海军', '官方网站', '消息']

    corpus.append(tiyu)
    corpus.append(yule)
    corpus.append(jioayu)
    corpus.append(shizheng)

    classVec = ['体育','娱乐','教育','时政']
    return  corpus,classVec



if __name__=='__main__':
    # corpus参数样例数据如下：
    corpus,classVec = loadDataSet()
    gensim_Corpus(corpus)
