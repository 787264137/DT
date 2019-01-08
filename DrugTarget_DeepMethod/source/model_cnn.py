import os
import time
import random as rn
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dropout, Activation
from keras.layers import Input, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model
from datahelper import DataSet
from utils import argparser, logging
from metrics import ACC
import matplotlib.pyplot as plt
from copy import deepcopy

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)  # 只要seed的值一样，后续生成的随机数都一样。
rn.seed(1)
tf.set_random_seed(0)

my_gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
# tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,  # 控制运算符op内部的并行
    inter_op_parallelism_threads=1,  # 控制多个运算符op之间的并行计算
    log_device_placement=False,  # 打印出各个操作在那个设备上运行
    gpu_options=my_gpu_options
)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

TABSY = "\t"


def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')  ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=128, input_length=FLAGS.max_smi_len)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=FLAGS.charseqset_size + 1, output_dim=128, input_length=FLAGS.max_seq_len)(
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
    predictions = Dense(FLAGS.num_classes, kernel_initializer='normal')(FC2)
    # OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='binary_crossentropy',
                             metrics=['accuracy'])  # , metrics=['acc_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='../data/figures/build_combined_categorical.png')

    return interactionModel


def nfold_1_2_3_setting_sample(data_6_folds, measure, runmethod, FLAGS, dataset):
    """
    通过表现最好的模型参数得到测试集的损失和精度
    :param XD:
    :param XT:
    :param Y:
    :param label_row_inds:
    :param label_col_inds:
    :param measure:
    :param runmethod:
    :param FLAGS:
    :param dataset:
    :return:
    """
    bestparamlist = []
    test_set, outer_train_sets = data_6_folds[0], data_6_folds[1:]
    num_train_folds = len(outer_train_sets)
    print("test set", str(test_set.iloc[:, 0].size))

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    # logger.info('Start training')
    for val_foldind in range(num_train_folds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = pd.concat(otherfolds,ignore_index=True)  # 把其他的训练集合并成一个训练集
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(val_fold.iloc[:,0].size))
        print("train set", str(otherfoldsinds.iloc[:,0].size))

    # 仅仅用作交叉验证选取在验证集上表现好的模型参数，all_predictions 和 all_losses不需要
    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(measure,
                                                                                                          runmethod,
                                                                                                          FLAGS,
                                                                                                          train_sets,
                                                                                                          val_sets,dataset)  # 验证集
    # 得到测试集在各个模型参数下的all_predictions 和 all_losses
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(measure, runmethod, FLAGS,
                                                                                         train_sets, test_sets,dataset)
    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)

    testperfs = []
    testloss = []
    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance ACC", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Loss", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd


def general_nfold_cv(prfmeasure, runmethod, FLAGS, labeled_sets, val_sets,dataset):  ## BURAYA DA FLAGS LAZIM????
    """
    交叉验证，选取在训练集训练好，在验证集上表现好的模型参数
    :param XD:
    :param XT:
    :param Y:
    :param label_row_inds:
    :param label_col_inds:
    :param prfmeasure:
    :param runmethod:
    :param FLAGS:
    :param labeled_sets:
    :param val_sets:
    :return:
    """
    paramset1 = FLAGS.num_windows  # [32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths  # [4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths  # [8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch  # 100
    batchsz = FLAGS.batch_size  # 256

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]  # 获取交叉验证的验证集
        labeledinds = labeled_sets[foldind]  # 获取交叉验证的训练集

        params = {}

        train_drugs, train_prots, train_Y = dataset.prepare_interaction_pairs(labeledinds)
        val_drugs, val_prots, val_Y = dataset.prepare_interaction_pairs(valinds)

        print(str(train_Y[0:100]))
        print(type(train_Y[0]))
        print(str(train_drugs[0]))
        print(type(train_drugs[0]))
        pointer = 0
        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)):  # learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]

                    gridmodel = runmethod(FLAGS, param1value, param2value, param3value)
                    gridres = gridmodel.fit(([np.array(train_drugs), np.array(train_prots)]), np.array(train_Y),
                                            batch_size=batchsz, epochs=epoch,
                                            validation_data=(
                                                ([np.array(val_drugs), np.array(val_prots)]), np.array(val_Y)),
                                            shuffle=False)

                    predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots)])
                    loss, rperf2 = gridmodel.evaluate(([np.array(val_drugs), np.array(val_prots)]), np.array(val_Y),
                                                      verbose=0)
                    rperf = prfmeasure(val_Y, predicted_labels)
                    rperf = rperf[0]

                    logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" %
                            (param1ind, param2ind, param3ind, foldind, rperf, rperf2, loss), FLAGS)

                    plotLoss(gridres, param1ind, param2ind, param3ind, foldind)

                    all_predictions[pointer][foldind] = rperf  # TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][foldind] = loss

                    pointer += 1

    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):
            for param3ind in range(len(paramset3)):

                avgperf = 0.
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= len(val_sets)
                # print(epoch, batchsz, avgperf)
                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1ind, param2ind, param3ind]

                pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def plotLoss(history, batchind, epochind, param3ind, foldind):
    figname = "b" + str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_" + str(foldind) + "_" + str(
        time.time())
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['trainloss', 'valloss', 'acc', 'valacc'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/" + figname + ".png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()

    ## PLOT acc
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.plot(history.history['acc_score'])
    plt.plot(history.history['val_acc_score'])
    plt.legend(['trainacc', 'valacc'], loc='upper left')
    plt.savefig("figures/" + figname + "_acc.png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)


def experiment(FLAGS, perfmeasure, deepmethod, setting_no):  # 5-fold cross validation + test
    # Input
    # XD: [drugs, features] sized array (features may also be similarities with other drugs
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    # perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    # higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    # foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation

    dataset = DataSet(fpath=FLAGS.dataset_path,  ### BUNU ARGS DA GUNCELLE
                      setting_no=FLAGS.setting_no,  ##BUNU ARGS A EKLE
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    data_6_folds = dataset.read_dataset(fpath=FLAGS.dataset_path, setting_no=setting_no)

    if not os.path.exists(FLAGS.fig_dir):
        os.makedirs(FLAGS.fig_dir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(data_6_folds, perfmeasure, deepmethod, FLAGS,
                                                                    dataset)

    logging("Setting Negative Sample" + str(FLAGS.setting_no), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_classification(FLAGS):
    """
    执行分类的函数
    :param FLAGS: utils中对于终端Arguments的配置
    :return:
    """
    perfmeasure = ACC
    deepmethod = build_combined_categorical

    experiment(FLAGS, perfmeasure, deepmethod, FLAGS.setting_no)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_classification(FLAGS)
