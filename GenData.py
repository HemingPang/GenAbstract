# """
# 预处理后，将两个语料合成gensim能接受的数据格式
# """
import codecs
import jieba.analyse
import os
import sys
from tqdm import tqdm
from tools.Common import get_num_lines

jieba.add_word('沪指')
jieba.add_word('沪深')
jieba.add_word('倒锤')


def remove_stopwords(tokens):
    if tokens is not None:
        return [token for token in tokens if token not in stopwords_set]


def gen_corpus(file):
    # f = codecs.open(file, 'r', 'utf-8')
    # line = f.readline()
    # while line:
    with open(file, "r", encoding='utf-8') as corpus:
        for line in tqdm(corpus, total=get_num_lines(file), desc='语料处理'):
            corpus_file.writelines(" ".join(remove_stopwords(jieba.cut(line))))


wiki_file, news_file, stopwords_file, output_file = sys.argv[1:5]

files = [wiki_file, news_file]

# 初始化停用词字典
stopwords_set = set()
with open(stopwords_file, 'r+', encoding='utf-8') as stopwords:
    for stpwd in stopwords:
        stopwords_set.add(stpwd.strip())

corpus_file = codecs.open(output_file, 'w', encoding='utf-8')
for file in files:
    gen_corpus(file)

corpus_file.close()

