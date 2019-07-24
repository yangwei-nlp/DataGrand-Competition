import pandas as pd
import numpy as np

train_set = pd.read_csv('data/train_set.csv', index_col=0)
print(train_set)
train_set.shape
train_set.iloc[0,]
train_set.columns
np.unique(train_set['class'])  # 19个类别

train_set['word_seg']


temp = train_set['word_seg'].map(lambda sentence: sentence.split())
sentences = [[int(word) for word in sent] for sent in temp.to_list()]
# memory error