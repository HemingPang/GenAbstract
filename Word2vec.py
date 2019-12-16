import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

logger.info("running %s" % ' '.join(sys.argv))

corpus_path = '/Users/ever/Documents/AI/NLP课程/projects/1/corpus/corpus.txt'
model_output = '/Users/ever/Documents/AI/NLP课程/projects/1/GenAbstract/result'
w2v_output = model_output

model = Word2Vec(sentences=LineSentence(source=corpus_path), size=400, window=5, min_count=5,
                 workers=multiprocessing.cpu_count())
model.save(model_output)
model.wv.save_word2vec_format(w2v_output, binary=False)
