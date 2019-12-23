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


class GenAbstract:
    def __init__(self, model_name):
        wiki_model = Word2Vec.load(model_name)
        self.w2v = wiki_model.wv
        self.embedding_size = self.w2v.vector_size
        self.vocab = self.w2v.vocab
        self.vocab_size = len(self.vocab)

    def get_word_frequency(self, word_text):
        if word_text in self.vocab:  # vocab是一个dict
            return self.vocab[word_text].count / self.vocab_size
        else:
            return 1e-6

    def sentence_to_vec(self, sentence_list: List[Sentence], a: float = 1e-3):
        """sentence_to_vec方法就是将句子转换成对应向量的核心方法"""
        sentence_set = []
        for sentence in sentence_list:
            sentence_length = sentence.len()
            if sentence_length == 0:
                continue
            vs = np.zeros(self.embedding_size)  # add all word2vec values into one vector for the sentence
            # 这个就是初步的句子向量的计算方法
            #################################################
            for word in sentence.word_list:
                a_value = a / (a + self.get_word_frequency(word.text))  # smooth inverse frequency, SIF
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
        if len(u) < self.embedding_size:
            for i in range(self.embedding_size - len(u)):
                # 列相加
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        sentence_vecs = []
        for vs in sentence_set:
            sub = np.multiply(u, vs)
            sentence_vecs.append(np.subtract(vs, sub))
        return sentence_vecs

    def get_word_list(self, sent) -> List[Word]:
        tokens_word = set(jieba.cut(sent))  # 去重
        return [Word(word, self.w2v[word]) for word in tokens_word if word in self.vocab]  # 词序是乱的，SIF最大软肋

    def get_relevance(self, v_s_i, v_c, v_t) -> float:
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

    def knn_smooth(self, sent_sim_dict: dict, window_size) -> dict:
        """根据window_size确定需要往前、后求多少个句子c_j的加权(算术)平均"""
        sents = list(sent_sim_dict.keys())
        sims = list(sent_sim_dict.values())
        sent_cluster_sim_dict = {}
        for idx, sent in enumerate(sents):
            idx_begin = 0 if idx - window_size < 0 else idx - window_size
            idx_end = len(sents) if idx_begin + window_size + 1 > len(sents) else idx_begin + window_size + 1
            key = "。".join(sents[idx_begin:idx_end])
            value = np.mean(sims[idx_begin:idx_end])
            sent_cluster_sim_dict[key] = value
        return sent_cluster_sim_dict

    def remove_overlap_and_keep_order(self, sents_top_n: List, tokens_sent: List):
        all_sents = "。".join(sents_top_n)
        sents_without_overlap = set(all_sents.split('。'))
        summary_in_order = ""
        for sent in tokens_sent:
            if sent in sents_without_overlap:
                summary_in_order += sent + "。"

        return summary_in_order

    def summarize(self, content, title):
        assert content is not None, 'Content is None'
        assert title is not None, 'Title is None'

        content_new = re.sub("\r\n", "", content)
        tokens_sent = content_new.split(
            '。')  # 文章的所有句子，todo: 疑问号、感叹号理论上也属于句子终结符号，是否也需要按照这些符号切割句子？我感觉不用，因为新闻类语料以这两个符号断句的情况较少
        logger.debug(tokens_sent)
        sentences_content = []  # 文章内容所有句子
        word_list_all = []
        for sent in tokens_sent:
            if sent is not None and sent:
                word_list = self.get_word_list(sent)
                word_list_all += word_list
                # logger.debug(word_list)
                sentences_content.append(Sentence(word_list))
        # logger.debug(json.dumps(sentences))
        logger.debug(', '.join([item.__str__() for item in sentences_content]))

        # 特殊：额外把title和全文所有内容各当做一个句子
        sentences_content.append(Sentence(word_list_all))  # 所有词组成一个长句
        sentences_content.append(Sentence(self.get_word_list(title)))  # title处理

        sent_vec = self.sentence_to_vec(sentences_content)
        v_t = sent_vec.pop()
        v_c = sent_vec.pop()
        v_s = sent_vec

        logger.debug('维度检查, v_s:%s, v_c:%s, v_t: %s' % (np.array(v_s).shape, np.array(v_c).shape, np.array(v_t).shape))

        # 得到每个句子的相关度dict
        sent_sim_dict = {item_sentence_.strip(): self.get_relevance(v_s_i, v_c=v_c, v_t=v_t) for (item_sentence_, v_s_i)
                         in
                         zip(tokens_sent, v_s)}

        top_n = 5
        # 没有平滑
        sents_top_n_without_smooth = dict(
            sorted(sent_sim_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
        logger.debug("没有平滑：%s" % sents_top_n_without_smooth)
        window_size = 1  # KNN平滑，前后各看1个句子，并计算其对应的加权平均
        sent_smooth_sim_dict = self.knn_smooth(sent_sim_dict, window_size)
        sents_top_n = dict(
            sorted(sent_smooth_sim_dict.items(), key=operator.itemgetter(1), reverse=True)[:top_n]).keys()
        # 消除overlap，并按照句子顺序输出
        summary = self.remove_overlap_and_keep_order(sents_top_n, tokens_sent)
        return summary

# if __name__ == 'main':
# model_name = 'result/wiki_1219.model'
# wiki_model = Word2Vec.load(model_name)
# w2v = wiki_model.wv
# embedding_size = w2v.vector_size
# vocab = w2v.vocab
# vocab_size = len(vocab)
#
# df = pd.read_csv('../corpus/sqlResult_1558435.csv', encoding="gb18030")
# title_content_df = df.loc[:, ['title', 'content']]
#
# idx = 7
# title_raw = title_content_df['title'][idx]
# text_raw = title_content_df['content'][idx]
# logger.debug("%s\n%s" % (title_raw, text_raw))
# logger.debug(summarize(content=text_raw, title=title_raw))
