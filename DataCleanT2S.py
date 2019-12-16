"""
wiki预料中部分为繁体，为了避免相同的词由于不同写法而导致分成不同的词，所以统一转换成简体。
此处使用opencc工具。Mac安装明朗：brew install opencc
"""

import os

dir = '/Users/ever/Documents/AI/NLP课程/projects/1/corpus/wiki_raw'
dir_out = '/Users/ever/Documents/AI/NLP课程/projects/1/corpus/wiki/'
sub_dirs = ['A' + chr(i) for i in range(65, 79)]
# print(sub_dirs)
output_file_index = 0
for sub_dir in sub_dirs:
    path = dir + "/" + sub_dir
    input_file_names = os.listdir(path)
    for name in input_file_names:
        input_file = path + "/" + name
        output_file_name = 'wiki_' + str(output_file_index)
        cmd = 'opencc -i ' + input_file + ' -o ' + dir_out + output_file_name + ' -c t2s.json'
        os.system(cmd)
        output_file_index += 1
