# coding:utf-8
import keras
import pandas as pd
import numpy as np
from datahelper import DataSet
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dropout
from keras.layers import Input, Embedding, Dense
from keras.models import Model

NUM_FILTERS = 100#128
dataset_path = '../data/mmc1_Processed/'
FILTER_LENGTH1 = 4
FILTER_LENGTH2 = 4
max_smi_len = 100
max_seq_len = 500
# 数据集
dataset = DataSet(fpath=dataset_path, setting_no=1, seqlen=500, smilen=100, need_shuffle=False)

charsmiset_size = dataset.charsmiset_size
charseqset_size = dataset.charseqset_size

data_6_folds = dataset.read_dataset(fpath=dataset_path, setting_no=1)
train_data, test_data, val_data = pd.concat(data_6_folds[2:], ignore_index=True), data_6_folds[0], data_6_folds[1]
# test_set, outer_train_sets = data_6_folds[0], data_6_folds[1:]
# num_train_folds = len(outer_train_sets)
# train_sets,val_sets,test_sets = []
#
# for val_foldind in range(num_train_folds):
#     val_fold = outer_train_sets[val_foldind]
#     val_sets.append(val_fold)
#     otherfolds = deepcopy(outer_train_sets)
#     otherfolds.pop(val_foldind)
#     otherfoldsinds = pd.concat(otherfolds, ignore_index=True)  # 把其他的训练集合并成一个训练集
#     train_sets.append(otherfoldsinds)
#     test_sets.append(test_set)
#     print("val set", str(val_fold.iloc[:, 0].size))
#     print("train set", str(otherfoldsinds.iloc[:, 0].size))
#
#
# for foldind in range(num_train_folds):
#     valinds = val_sets[foldind]  # 获取交叉验证的验证集
#     labeledinds = train_sets[foldind]  # 获取交叉验证的训练集
#
#     params = {}

train_drugs, train_prots, train_Y = dataset.prepare_interaction_pairs(train_data)
val_drugs, val_prots, val_Y = dataset.prepare_interaction_pairs(val_data)
test_drugs, test_prots, test_Y = dataset.prepare_interaction_pairs(test_data)

train_drugs, train_prots, train_Y = np.array(train_drugs), np.array(train_prots), np.array(train_Y)
val_drugs, val_prots, val_Y = np.array(val_drugs), np.array(val_prots), np.array(val_Y)
test_drugs, test_prots, test_Y = np.array(test_drugs), np.array(test_prots), np.array(test_Y)

# 探索数据
print('Training entries:{},lables:{}'.format(len(train_drugs), len(train_Y)))

print(train_drugs[0])

# 构建模型
XDinput = Input(shape=(max_smi_len,), dtype='int32')  ### Buralar flagdan gelmeliii
XTinput = Input(shape=(max_seq_len,), dtype='int32')

### SMI_EMB_DINMS  FLAGS GELMELII
encode_smiles = Embedding(input_dim=charsmiset_size + 1, output_dim=128, input_length=max_smi_len)(
    XDinput)
encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                       strides=1)(encode_smiles)
encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                       strides=1)(encode_smiles)
encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                       strides=1)(encode_smiles)
encode_smiles = GlobalMaxPooling1D()(encode_smiles)

encode_protein = Embedding(input_dim=charseqset_size + 1, output_dim=128, input_length=max_seq_len)(
    XTinput)
encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                        strides=1)(encode_protein)
encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                        strides=1)(encode_protein)
encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                        strides=1)(encode_protein)
encode_protein = GlobalMaxPooling1D()(encode_protein)

encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1)

# Fully connected
FC1 = Dense(1024, activation='relu')(encode_interaction)
FC2 = Dropout(0.1)(FC1)
FC2 = Dense(1024, activation='relu')(FC2)
FC2 = Dropout(0.1)(FC2)
FC2 = Dense(512, activation='relu')(FC2)

# And add a logistic regression on top
predictions = Dense(1, kernel_initializer='normal')(FC2)
# OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

interactionModel.compile(optimizer='adam', loss='binary_crossentropy',
                         metrics=['accuracy'])  # , metrics=['acc_score']
print(interactionModel.summary())

# 训练模型
history = interactionModel.fit([train_drugs, train_prots], train_Y,
                               epochs=40, batch_size=512,
                               validation_data=([val_drugs, val_prots], val_Y),
                               verbose=1)

results = interactionModel.evaluate([test_drugs, test_prots], test_Y)
print(results)
