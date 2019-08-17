"""
Description :   将预处理后的数据集变换为BiLSTM+CRF模型需要的输入格式
     Author :   Yang
       Date :   2019/8/17
"""
import codecs


training_data = []
word_collections = []  # 词典
tag_to_ix = {}  # 标注集
with codecs.open('data/dg_train.txt', 'r', encoding='utf-8') as f:
    # f = codecs.open('data/dg_train.txt', 'r', encoding='utf-8')
    line = f.readline()
    features = []  # 每句话的所有词
    tags = []  # 每句话中每个词的标注
    while line:
        if line == '\n':
            training_data.append((features, tags))
            features = []
            tags = []
        else:
            word = int(line.split('\n')[0].split('\t')[0])
            tag = line.split('\n')[0].split('\t')[1]
            if word not in word_collections:
                word_collections.append(word)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
            features.append(word)
            tags.append(tag)
        line = f.readline()
vocab_size = len(word_collections)
