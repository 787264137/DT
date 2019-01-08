# coding:utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.data_utils import get_file
import json

"""
药物靶点相互作用分类
"""
# 下载IMDB数据集
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 探索数据
print('Training entries:{},lables:{}'.format(len(train_data), len(train_labels)))

print(train_data[0])
print(len(train_data[0]), len(train_data[1]), len(train_data[2]))


# 获取对于字词的编码
# encode
def get_word_index(path):
    """Retrieves the dictionary mapping word indices back to words.

    Arguments:
        path: where to cache the data (relative to `~/.keras/dataset`).

    Returns:
        The word index dictionary.
    """
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = get_file(
        path,
        origin=origin_folder + 'imdb_word_index.json',
        file_hash='bfafd718b763782e994055a2d397834f')
    with open(path) as f:
        return json.load(f)


word_index = get_word_index('imdb_word_index.json')

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# 从句子的编码解析出句子
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i) for i in text])


print(decode_review(train_data[0]))

# 由于影评的长度必须相同，我们将使用 pad_sequences 函数将长度标准化：
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
print(train_data[0])

# 构建模型
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))  # 该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。模型在接受训练时会学习这些向量。
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# 损失函数和优化器
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

#创建验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

model.summary()
# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data,test_labels)
print(results)

