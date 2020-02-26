from tensorflow.keras.preprocessing import text
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import swifter
import os

# https://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%BB%93%E6%9E%9C%E6%8F%90%E4%BA%A4.html
# https://blog.csdn.net/m0_37306360/article/details/85030283

# os.chdir(r'D:\..courses\DataGrand-Competition\2018-competition')

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 指定GPU

modelpath = "word2vec.model"

# 1.训练所有词的词向量
if not os.path.exists(modelpath):
    train_set = pd.read_csv('data/train_set.csv', index_col=0)
    # 注意不能在测试集训练，因为测试成绩是AB榜形式
    corpus = train_set['word_seg'].swifter.apply(lambda line: line.split())

    w2c_model = Word2Vec(corpus.values.tolist(), size=200, window=6, min_count=20,
                         iter=30, workers=cpu_count())
    w2c_model.save(modelpath)
else:
    w2c_model = Word2Vec.load(modelpath)


# 2.构建模型
def my_model(max_len, embedding_matrix):
    content = keras.layers.Input(shape=(max_len,))
    embedding = keras.layers.Embedding(input_dim=embedding_matrix.shape[0],
                                       weights=[embedding_matrix],
                                       output_dim=embedding_matrix.shape[1],
                                       trainable=False)
    x = keras.layers.SpatialDropout1D(0.1)(embedding(content))

    x = keras.layers.Bidirectional(keras.layers.GRU(200, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(200, return_sequences=True))(x)

    avg_pool = keras.layers.GlobalAveragePooling1D()(x)
    max_pool = keras.layers.GlobalMaxPooling1D()(x)

    conc = tf.concat([avg_pool, max_pool], axis=-1)

    x = keras.layers.Dropout(0.2)(
        keras.layers.Activation(activation="relu")(keras.layers.BatchNormalization()(keras.layers.Dense(1000)(conc))))
    x = keras.layers.Activation(activation="relu")(keras.layers.BatchNormalization()(keras.layers.Dense(500)(x)))
    output = keras.layers.Dense(19, activation="softmax")(x)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = keras.models.Model(inputs=content, outputs=output)
    model.compile(loss=loss_object, optimizer='adam', metrics=['accuracy'])
    return model


# 3.保存部分词向量、并拟合数据集
train_set = pd.read_csv('data/train_set.csv', index_col=0)
test_set = pd.read_csv('data/test_set.csv', index_col=0)

train_set = train_set.dropna(axis=0, how='any')
test_set = test_set.dropna(axis=0, how='any')

tokenizer = text.Tokenizer(num_words=250000, lower=False, filters="")
tokenizer.fit_on_texts(train_set['word_seg'].values.tolist() +
                       test_set['word_seg'].values.tolist())

embedding_matrix = np.zeros((len(tokenizer.index_word) + 1, 200))
for i, word in tokenizer.index_word.items():
    word_vector = w2c_model[word] if word in w2c_model else None
    if word_vector is not None:
        embedding_matrix[i] = word_vector
    else:
        unk_vec = np.random.random(200) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_matrix[i] = unk_vec

trains = keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(train_set['word_seg'].values), maxlen=256
)

tests = keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(test_set['word_seg'].values), maxlen=256
)

trainings = tf.data.Dataset.from_tensor_slices((trains, train_set['class'] - 1))
trainings = trainings.batch(256)

testings = tf.data.Dataset.from_tensor_slices((tests))
testings = testings.batch(256)

checkpoint_dir = "model/cp-{epoch:04d}.ckpt"
callbacks = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                               verbose=1,
                                               save_weights_only=False,
                                               period=10)

model = my_model(256, embedding_matrix)

if os.path.exists("model"):
    latest = tf.train.latest_checkpoint("model")
    model.load_weights(latest)
    print('Model Checkpoint Loaded!!')
else:
    model.save_weights(checkpoint_dir.format(epoch=0))


if __name__ == "__main__":
    model.fit(trainings, epochs=4, callbacks=[callbacks])

else:
    predictions = model.predict(testings)
    np.argmax(predictions, axis=1)
