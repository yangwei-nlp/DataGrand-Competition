from gensim.test.utils import common_texts, get_tmpfile
import tensorflow.keras.preprocessing.text as T
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from tensorflow import keras
import pandas as pd
import numpy as np
import swifter
import os

#os.chdir(r'D:\..courses\DataGrand-Competition\2018-competition')

modelpath = "word2vec.model"

# 1.获得词向量
if not os.path.exists(modelpath):
    train_set = pd.read_csv('data/train_set.csv', index_col=0)
    corpus = train_set['word_seg'].swifter.apply(lambda line: line.split())
    tokenizer = T.Tokenizer(num_words=200000, lower=False, filters="")
    [tokenizer.fit_on_texts(sentence) for sentence in corpus]
    print(len(tokenizer.index_word))

    model = Word2Vec(corpus.values.tolist(), size=200, window=5, min_count=1,
                     iter=10, workers=cpu_count(), max_vocab_size=200000)
    model.save(modelpath)
else:
    Word2Vec.load(modelpath)

# model.wv.vectors
