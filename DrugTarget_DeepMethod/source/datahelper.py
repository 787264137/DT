import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#  Define CHARSET, CHARLEN

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25

CHARISOSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, "/": 7, ".": 8,
                 "1": 9, "0": 10, "3": 11, "2": 12, "5": 13, "4": 14, "7": 15, "6": 16,
                 "9": 17, "8": 18, "=": 19, "A": 20, "@": 21, "C": 22, "B": 23, "E": 24,
                 "D": 25, "G": 26, "F": 27, "I": 28, "H": 29, "K": 30, "M": 31, "L": 32,
                 "O": 33, "N": 34, "P": 35, "S": 36, "R": 37, "U": 38, "T": 39, "W": 40,
                 "V": 41, "Y": 42, "[": 43, "Z": 44, "]": 45, "\\": 46, "a": 47, "c": 48,
                 "b": 49, "e": 50, "d": 51, "g": 52, "f": 53, "i": 54, "h": 55, "m": 56,
                 "l": 57, "o": 58, "n": 59, "s": 60, "r": 61, "u": 62, "t": 63, "y": 64}

CHARISOSMILEN = 64


#  Encoding Helpers

#  Y = -(np.log10(Y/(math.pow(math.e,9))))

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)))  # +1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch] - 1)] = 1

    return X  # .tolist()


def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1

    return X  # .tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    """
    label encoding for smiles
    把smiles中的每一个元素通过数字词典转换成相应的数字
    :param line:
    :param MAX_SMI_LEN:
    :param smi_ch_ind:
    :return:
    """
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()


def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    """
    label encoding for sequence
    把蛋白质序列中的每一个元素通过数字词典转换成相应的数字
    :param line:
    :param MAX_SEQ_LEN:
    :param smi_ch_ind:
    :return:
    """
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()


#  DATASET Class
# works for large dataset
class DataSet(object):
    def __init__(self, fpath, setting_no, seqlen, smilen, need_shuffle=False):
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        # self.NCLASSES = n_classes
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET  ###HERE CAN BE EDITED
        self.charsmiset_size = CHARISOSMILEN
        self.PROBLEMSET = setting_no  # 交叉验证时的train，test的文件标签

    def read_dataset(self, fpath, setting_no):
        print("Reading dataset %s start" % fpath)

        # fpath = '/Users/stern/PycharmProjects/Stern/DTI/DrugTarget_DeepMethod/data/mmc1_Processed/'
        # setting_no = 1
        enzyme = self.get_positive_and_negative(fpath, 'Enzyme', setting_no)
        enzymes = self.split_n_folds(6, enzyme)
        gpcr = self.get_positive_and_negative(fpath, 'GPCR', setting_no)
        gpcrs = self.split_n_folds(6, gpcr)
        ionchannel = self.get_positive_and_negative(fpath, 'IonChannel', setting_no)
        ionchannels = self.split_n_folds(6, ionchannel)
        nuclearreceptor = self.get_positive_and_negative(fpath, 'NuclearReceptor', setting_no)
        nuclearreceptors = self.split_n_folds(6, nuclearreceptor)

        num = enzyme.iloc[:, 0].size + gpcr.iloc[:, 0].size + ionchannel.iloc[:, 0].size + nuclearreceptor.iloc[:,
                                                                                           0].size
        print("===数据集总共%d条数据===" % num)

        data_6_folds = [pd.concat([enzymes[i], gpcrs[i], ionchannels[i], nuclearreceptors[i]], ignore_index=True) for i
                        in range(6)]
        print(data_6_folds[0].iloc[:, 0].size)
        return data_6_folds

    def get_positive_and_negative(self, fpath, filename, setting_no):
        """
        获取正样本数据和负样本数据
        :param fpath:
        :param filename:
        :param setting_no:
        :return:
        """
        positive = pd.read_excel(fpath + filename + "_p/" + filename + ".xls")
        positive.loc[:, 'Y'] = 1
        negative = pd.read_excel(fpath + filename + "_p/" + filename + "decoy" + str(setting_no) + '.xls')
        negative.loc[:, 'Y'] = 0
        print('%s正样本有%d条数据，负样本有%d条数据' % (filename, positive.iloc[:, 0].size, negative.iloc[:, 0].size))
        # 删除蛋白质序列为空的行
        positive = positive.dropna(how='any')
        negative = negative.dropna(how='any')
        data = pd.concat([positive, negative], ignore_index=True)
        data = shuffle(data)
        return data

    def split_n_folds(self, n, df):
        """
        将数据框分为n份
        :param n:
        :param df:
        :return:
        """
        length = df.iloc[:, 0].size
        fold_num = int(length // n)
        dfs = []
        for i in range(n):
            dfs.append(df.iloc[i * fold_num:(i + 1) * fold_num, :])
        return dfs

    def prepare_interaction_pairs(self, df):
        """
        将数据框转化为DrugSmiles，TargetSequence，Y的格式,并对其进行label encoding
        :param df:
        :return:
        """
        print("Label encoding start")
        XD, XT, Y = [], [], []
        length = df.iloc[:, 0].size
        for i in range(length):
            smiles = df.loc[i, 'smiles']
            labeled_smiles = label_smiles(smiles, self.SMILEN, self.charsmiset)
            sequence = df.loc[i, 'sequence']
            # print(df.loc[i, 'drug'])
            # print(df.loc[i, 'protein'])
            # print(df.loc[i, 'sequence'])
            labeled_sequence = label_sequence(sequence, self.SEQLEN, self.charseqset)
            y = df.loc[i, 'Y']

            XD.append(labeled_smiles)
            XT.append(labeled_sequence)
            Y.append(y)
        return XD, XT, Y
