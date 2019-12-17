import sys
from gensim.models import Word2Vec
import logging
import os

program_name, model_name, test_word, should_test_linear = sys.argv[:4]
## logger设计
program = os.path.basename(program_name)
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
####

logger.info("running %s" % ' '.join(sys.argv))

wiki_model = Word2Vec.load(model_name)
result = wiki_model.wv.most_similar(test_word)
if result is not None:
    logger.info()

if int(should_test_linear):
    x1, x2, y1 = sys.argv[4:7]
    result = wiki_model.most_similar(positive=[y1, x2], negative=[x1])
    if result is not None:
        logger.info(result)
