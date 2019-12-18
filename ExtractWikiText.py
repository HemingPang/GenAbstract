import codecs
from gensim.corpora import WikiCorpus
import logging
from hanziconv import HanziConv
import sys

logger = logging.getLogger("ExtractWikiText")
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

wiki_file, wiki_zh_text = sys.argv[1:3]
# '/Users/ever/Documents/AI/NLP课程/projects/1/corpus/zhwiki-20191201-pages-articles.xml.bz2'
#  = 'data/wiki_zh_plain.txt'
wiki_corpus = WikiCorpus(wiki_file, dictionary={})

output_file = codecs.open(wiki_zh_text, 'w', encoding='utf-8')
texts = wiki_corpus.get_texts()
# logger.info(type(texts))

text_count = 0
# for text in tqdm(texts, total=len(texts), desc='获取wiki中文'):
for text in texts:
    raw_text = ' '.join(text)  # 包含繁体字，需要弄成简体
    to_simplified = HanziConv.toSimplified(raw_text)
    output_file.writelines(to_simplified + '\n')
    text_count += 1
    if text_count % 10000 == 0:
        logger.info('已处理%d 篇文章' % text_count)

output_file.close()
