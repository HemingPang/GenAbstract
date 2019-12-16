import pandas as pd
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',1000)

# df = pd.read_csv('sqlResult_1558435.csv',encoding = "ISO-8859-1")
df = pd.read_csv('sqlResult_1558435.csv',encoding = "gb18030")

# text = df['content'][7]
# print(text.strip())
# print(df['title'][4])
print(df['content'][4])
# print(df.tail())