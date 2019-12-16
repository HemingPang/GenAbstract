import pandas as pd
import numpy as np
import codecs
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv('sqlResult_1558435.csv', encoding="gb18030")
data = df['content']

file = codecs.open('news_corpus.txt', 'w', encoding='utf-8')

for idx, text in data.items():
    if not pd.isna(text):
        try:
            text = text.replace(u'\u3000', u'').replace(u'\\n', u'')
            sentences = text.split('\r\n')
            print(sentences)
            for line in sentences:
                if line:
                    file.write(line.strip() + '\r\n')
        except Exception:
            print('%s, %s, %s' % (idx, text, np.isnan(text)))

file.close()

# print(df.tail())
