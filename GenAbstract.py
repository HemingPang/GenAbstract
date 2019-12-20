import numpy as np
from sklearn.decomposition import PCA
from typing import List
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab
import pandas as pd
import re
import logging
import os
import jieba
import operator

program = os.path.basename("DataGlimpse")
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(funcName)s: %(lineno)d}: %(message)s')
# logging.root.setLevel(level=logging.INFO)
logging.root.setLevel(level=logging.DEBUG)

jieba.add_word('沪指')
jieba.add_word('沪深')
jieba.add_word('倒锤')


class Word:
    def __init__(self, text, vector: Vocab):
        self.text = text  # 文本
        self.vector = vector  # 词向量

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.text)


class Sentence:
    def __init__(self, word_list: List[Word]):
        self.word_list = word_list

    def len(self) -> int:
        return len(self.word_list)

    def __str__(self):
        vars = [item.__str__() for item in self.word_list if item is not None]
        return "{sentence: %s}" % ','.join(vars)

    # def _2_str(self):
    #     return "".join(self.word_list)


def get_word_frequency(word_text):
    if word_text in vocab:  # vocab是一个dict
        return vocab[word_text].count / vocab_size
    else:
        return 1e-6


def sentence_to_vec(sentence_list: List[Sentence], a: float = 1e-3):
    """sentence_to_vec方法就是将句子转换成对应向量的核心方法"""
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        # 这个就是初步的句子向量的计算方法
        #################################################
        for word in sentence.word_list:
            a_value = a / (a + get_word_frequency(word.text))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences
    #################################################
    # calculate PCA of this sentence set,计算主成分
    pca = PCA()
    # 使用PCA方法进行训练
    pca.fit(np.array(sentence_set))
    # 返回具有最大方差的的成分的第一个,也就是最大主成分,
    # components_也就是特征个数/主成分个数,最大的一个特征值
    u = pca.components_[0]  # the PCA vector
    # 构建投射矩阵
    u = np.multiply(u, np.transpose(u))  # u x uT
    # judge the vector need padding by whether the number of sentences less than embeddings_size
    # 判断是否需要填充矩阵,按列填充
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            # 列相加
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))
    return sentence_vecs


def get_word_list(sent) -> List[Word]:
    tokens_word = set(jieba.cut(sent))  # 去重
    return [Word(word, w2v[word]) for word in tokens_word if word in vocab]  # 词序是乱的，SIF最大软肋


def get_relevance(v_s_i, v_c, v_t) -> float:
    """获取某个句子与全文的相关度(0,1)"""
    v_s_i = np.array(v_s_i).reshape(-1, 400)  # v_c和v_t都变成了1x400的矩阵，所以要将切割出来的v_s_i从向量变成矩阵
    v_c = np.array(v_c).reshape(-1, 400)
    v_t = np.array(v_t).reshape(-1, 400)

    numerator_s_c = np.dot(v_s_i, np.transpose(v_c))  #
    denominator_s_c = np.linalg.norm(v_s_i) * np.linalg.norm(v_c)
    cos_s_c = numerator_s_c / denominator_s_c  # 范围是[-1,1]
    similarity_s_c = 0.5 + 0.5 * cos_s_c  # 归一化 范围[0,1]

    # 获取某个句子与标题的相关度(0,1)
    numerator_s_t = np.dot(v_s_i, np.transpose(v_t))
    denominator_s_t = np.linalg.norm(v_s_i) * np.linalg.norm(v_t)
    cos_s_t = numerator_s_t / denominator_s_t  # 范围是[-1,1]
    similarity_s_t = 0.5 + 0.5 * cos_s_t  # 归一化 范围[0,1]
    # todo 理论上标题的权重要大点
    return np.mean([similarity_s_c, similarity_s_t])


def summarize(content, title):
    assert content is not None, 'Content is None'
    assert title is not None, 'Title is None'

    content_new = re.sub("\r\n", "", content)
    tokens_sent = content_new.split('。')  # 文章的所有句子，todo: 疑问号、感叹号理论上也属于句子终结符号，是否也需要按照这些符号切割句子？我感觉不用，因为新闻类语料以这两个符号断句的情况较少
    logger.debug(tokens_sent)
    sentences_content = []  # 文章内容所有句子
    word_list_all = []
    for sent in tokens_sent:
        if sent is not None and sent:
            word_list = get_word_list(sent)
            word_list_all += word_list
            # logger.debug(word_list)
            sentences_content.append(Sentence(word_list))
    # logger.debug(json.dumps(sentences))
    logger.debug(', '.join([item.__str__() for item in sentences_content]))

    # 特殊：额外把title和全文所有内容各当做一个句子
    sentences_content.append(Sentence(word_list_all))  # 所有词组成一个长句
    sentences_content.append(Sentence(get_word_list(title)))  # title处理

    sent_vec = sentence_to_vec(sentences_content)
    v_t = sent_vec.pop()
    v_c = sent_vec.pop()
    v_s = sent_vec

    logger.debug('维度检查, v_s:%s, v_c:%s, v_t: %s' % (np.array(v_s).shape, np.array(v_c).shape, np.array(v_t).shape))

    # 得到每个句子的相关度dict
    sent_sim_dict = {item_sentence_: get_relevance(v_s_i, v_c=v_c, v_t=v_t) for (item_sentence_, v_s_i) in
                     zip(tokens_sent, v_s)}
    # 返回top 5
    top_n = 5
    sents_top_n = dict(sorted(sent_sim_dict.items(), key=operator.itemgetter(1), reverse=True)[:top_n]).keys()
    summary = '。'.join(sents_top_n)
    return summary


model_name = 'result/wiki_1219.model'
wiki_model = Word2Vec.load(model_name)
w2v = wiki_model.wv
embedding_size = w2v.vector_size
vocab = w2v.vocab
vocab_size = len(vocab)

df = pd.read_csv('../corpus/sqlResult_1558435.csv', encoding="gb18030")
title_content_df = df.loc[:, ['title', 'content']]

title_1 = '增值税消费税等立法密集启幕 明年税收立法将再提速'
text = '“落实税收法定原则”，首次明确提出是在2013年，党的十八届三中全会通过的《中共中央关于全面深化改革若干重大问题的决定》中提出。' \
       'AAAA按照这一原则，新开征的税种应制定相应的税收法律，现行的税收条例应通过修改上升为法律AAAA。近年来，我国多项财政法律、法规制定工作蹄疾步稳。尤其是自2018年下半年以来，' \
       '几乎每次人大常委会会议都有相关税收法律审议内容，税收法定步入快车道。落实税收法定原则，2019年是关键的一年、攻坚的一年。在这一年，通过了一部税法——资源税法，此外，' \
       '包括土地增值税法、增值税法、消费税法在内的多部税法公布了财政部部内征求意见稿，契税、城建税等将提交全国人大常委会审议。2019年8月26日，经十三届全国人大常委会第十二次会议表决，' \
       '资源税法正式通过，将于2020年9月1日正式实施。日前，继土地增值税法征求意见稿发布后，财政部陆续公布了增值税法、消费税法的征求意见稿。尤其是作为我国第一大税种的增值税立法迈出实质性步伐，' \
       '被业内视作我国落实税收法定的关键一步，对于健全我国现代税收制度体系具有积极的意义。中国政法大学教授、财税法研究中心主任施正文对《经济参考报》记者表示，体量上来说，增值税是我国第一大税种，' \
       '其立法是我国落实税收法定原则的重大进步。@@@@对于2020年落实税收法定，增值税能否立法起到至关重要的作用。@@@@ #######北京国家会计学院财税政策与应用研究所所长李旭红对《经济参考报》记者表示，为落实税收法定原则，' \
       '2019年完成多个税种的立法工作，统筹考虑、同步推进，及时让税制改革成果体现为法律，通过落实税收法定来进一步巩固及拓展减税降费的成效，也为2020年全面落实税收法定原则打下坚实基础。#######除了落实税收法定之外，' \
       '2019年的税收立法保障了减税降费成果，增强了税收确定性$$$$$$$$$。李旭红指出，随着2019年税收制度加快落实税收立法，一些短期性的政策性减免将转变为法定式减免，通过落实税收法定来巩固和拓展减税降费成效，' \
       '将深化税制改革与落实税收法定原则相结合，及时以法律形式体现税制改革成果，有效增强税收确定性，从而形成了协同作用，在振兴实体经济、激发市场活力、助力创新创业、推动高质量发展方面发挥了重要作用。&&&&&&&&' \
       '中国法学会财税法学研究会会长、北京大学法学院教授刘剑文指出，税收涉及千家万户的切身利益，税收立法既是为了落实税收法定原则，同时也是推进国家治理现代化和民主法治建设的重要内容。' \
       '当前加强立法、补上财税立法的短板，将税收纳入法治框架意义重大。同时，税收法定的落实是不断形成共识的过程，是牵一发而动全身的艰巨复杂工作，每一项都要稳妥推进，植根我国现实国情，既立足现阶段，也着眼长远。'
logger.debug(summarize(text, title_1))

'''北京国家会计学院财税政策与应用研究所所长李旭红对《经济参考报》记者表示，为落实税收法定原则，2019年完成多个税种的立法工作，统筹考虑、同步推进，及时让税制改革成果体现为法律，
通过落实税收法定来进一步巩固及拓展减税降费的成效，也为2020年全面落实税收法定原则打下坚实基础。除了落实税收法定之外，2019年的税收立法保障了减税降费成果，增强了税收确定性。
李旭红指出，随着2019年税收制度加快落实税收立法，一些短期性的政策性减免将转变为法定式减免，通过落实税收法定来巩固和拓展减税降费成效，将深化税制改革与落实税收法定原则相结合，及时以法律形式体现税制改革成果，
有效增强税收确定性，从而形成了协同作用，在振兴实体经济、激发市场活力、助力创新创业、推动高质量发展方面发挥了重要作用。对于2020年落实税收法定，增值税能否立法起到至关重要的作用。
按照这一原则，新开征的税种应制定相应的税收法律，现行的税收条例应通过修改上升为法律'''
