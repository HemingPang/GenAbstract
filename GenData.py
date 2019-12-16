# """
# 预处理后，将两个语料合成gensim能接受的数据格式
# """
import codecs
import jieba.analyse
import os
import threading


class WriteCorpusThread(threading.Thread):
    def __init__(self, files, output_file):
        threading.Thread.__init__(self)
        self.files = files
        self.output_file = output_file

    def run(self) -> None:
        for file in files:
            self.gen_corpus(file)

    def gen_corpus(self, file):
        f = codecs.open(file, 'r', 'utf-8')
        line = f.readline()
        corpus_file = codecs.open('../corpus/' + self.output_file, 'w', encoding='utf-8')
        while line:
            corpus_file.writelines(" ".join(jieba.cut(line)))
            line = f.readline()
        f.close()
        corpus_file.close()


wiki_path = '/Users/ever/Documents/AI/NLP课程/projects/1/corpus/wiki'
news_file = '/Users/ever/Documents/AI/NLP课程/projects/1/corpus/news_corpus_raw.txt'

files = [wiki_path + '/' + file_name for file_name in os.listdir(wiki_path)]
files.append(news_file)

# corpus_file = codecs.open('../corpus/corpus.txt', 'w', encoding='utf-8')

# 10个线程
block_size = int(len(files) / 10)
block_files = [files[i:i + block_size] for i in range(0, len(files), block_size)]

block_idx = 0
for temp_files in block_files:
    write_corpus_thread = WriteCorpusThread(temp_files, 'corpus_' + str(block_idx) + '.txt')
    write_corpus_thread.start()
    block_idx += 1

# corpus_file.close()
