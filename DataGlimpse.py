import pandas as pd
import logging
import os

program = os.path.basename("DataGlimpse")
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(funcName)s: %(lineno)d}: %(message)s')
logging.root.setLevel(level=logging.INFO)
####

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 1000)

# df = pd.read_csv('sqlResult_1558435.csv',encoding = "ISO-8859-1")
df = pd.read_csv('../corpus/sqlResult_1558435.csv', encoding="gb18030")

logger.info(df.columns)

target_df = df.loc[:,['title','content']]
logger.info(target_df.head(1))
# article = df.head(2)
# logger.info(article)
# logger.info(type(article['id']))
# logger.info('id: %s' % article['id'].get(0))
# logger.info('author: %s' % article['author'][0])
# logger.info('content: %s' % article['content'][0])
# # logger.info('feature: %s' % article['feature'][0])
# logger.info('title: %s' % article['title'][0])
# text = df['content'][7]
# print(text.strip())
# print(df['title'][4])
# print(df['content'][4])
# print(df.tail())
