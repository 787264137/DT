# coding:utf-8
import numpy as np
import pandas as pd
from pychem import fingerprint, pychem
import os



def mergeDF(columns, *dfs):
    """

    :param columns: list
    :param dfs: cuple of dataFrame
    :return:
    """
    if len(columns) != len(dfs):
        raise Exception('列参数不足')
    d = {}
    for i in range(len(columns)):
        # print(type(dfs[i]))
        if type(dfs[i]) != pd.Series:
            raise Exception('数据类型必须为Series')
        d[columns[i]] = dfs[i]
    merged = pd.DataFrame(d)
    return merged


def getIDs(file):
    df = pd.read_excel(file)
    drugIds = df.iloc[:, 1]
    proteinIds = df.iloc[:, 0]
    # print(drugIds.head())
    # print(proteinIds.head())
    return drugIds, proteinIds


def getDrugDict(file):
    """

    :param file:
    :return: drugDict,key:Drug value:smile
    """
    df = pd.read_excel(file)
    drugDict = {}
    rowNum = len(df.iloc[:,0])
    for row in range(rowNum):
        drugDict[df.values[row][0]] = df.values[row][1]
    return drugDict


def getProteinDict(file):
    """

    :param file:
    :return: proteinDict,key:protein value:sequence
    """
    df = pd.read_excel(file)
    proteinDict = {}
    rowNum = len(df.iloc[:,0])
    for row in range(rowNum):
        proteinDict[df.values[row][0]] = df.values[row][1]
    return proteinDict


def getDrugSmilesAndMacc(drugIds, drugDict):
    count = 0
    rowNum = len(drugIds)
    df_smilesAndMacc = pd.DataFrame(np.zeros([rowNum, 2]), columns=['smiles', 'macc'])
    for row in range(rowNum):
        key = drugIds[row]
        if key in drugDict:
            smiles = drugDict[key]
            mol = pychem.Chem.MolFromSmiles(smiles)
            res = fingerprint.CalculateMACCSFingerprint(mol)
            fp = ''
            for i in range(res[0]):
                if i in res[1]:
                    fp += '1'
                else:
                    fp += '0'

            df_smilesAndMacc.loc[row, 'smiles'] = smiles
            df_smilesAndMacc.loc[row, 'macc'] = fp
        else:
            count += 1
    print('总共有%d条药物数据,其中有%d条药物结构缺失。' % (rowNum, count))
    return df_smilesAndMacc


def getProteinSequence(proteinIds, proteinDict):
    count = 0
    rowNum = len(proteinIds)
    df_proteinSequence = pd.DataFrame(np.zeros([rowNum, 1]), columns=['sequence'])
    for row in range(rowNum):
        key = proteinIds[row]
        key = ''.join(key.split(':'))  # 处理pair文件中的靶点名称hsa与数字间多了个分号
        if key in proteinDict:
            df_proteinSequence.loc[row, 'sequence'] = proteinDict[key]
        else:
            count += 1
    print("总共有%d条蛋白质数据，其中有%d条蛋白质序列缺失。" % (rowNum, count))
    # print(type(df_proteinSequence.loc[:,'sequence']))
    return df_proteinSequence.loc[:,'sequence']


def writeToFile(dfs, outPath):
    dfs.to_excel(outPath, encoding='utf-8', index=False)


def processOneFile(columns,pairPath, drugDictPath, proteinDictPath, outFilePath):
    drugIds, proteinIds = getIDs(pairPath)
    drugDict, proteinDict = getDrugDict(drugDictPath), getProteinDict(proteinDictPath)
    dfDrugSmilesAndMacc = getDrugSmilesAndMacc(drugIds, drugDict)
    dfDrugMacc = dfDrugSmilesAndMacc.loc[:, 'macc']
    dfProteinSequence = getProteinSequence(proteinIds, proteinDict)
    dfs = mergeDF(columns,drugIds, proteinIds, dfDrugMacc, dfProteinSequence)
    writeToFile(dfs, outFilePath)
    print('==========================')
    print('%s文件处理完毕，处理后的文件写入到%s' % (os.path.split(pairPath)[1], outFilePath))


def getDirList(p):
    p = str(p)
    if p == '':
        return []
    if p[-1] != '/':
        p = p + '/'
    a = os.listdir(p)
    b = [x for x in a if os.path.isdir(p + x)]

    return b


def getFileList(p):
    p = str(p)
    if p == '':
        return []
    if p[-1] != '/':
        p = p + '/'
    a = os.listdir(p)
    b = [x for x in a if os.path.isfile(p+x)]

    return b


if __name__ == '__main__':
    rootDirectory = '/Users/stern/Desktop/Dataset/LargeScale/mmc1'  # 数据根目录
    drugDictPath = '/Users/stern/Desktop/Dataset/LargeScale/DictFile/Drug_smi'
    proteinDictPath = '/Users/stern/Desktop/Dataset/LargeScale/DictFile/protein_hsa'
    outPath = '/Users/stern/Desktop/Dataset/LargeScale/mmc1_Processed'

    columns = ['drug','protein','macc','sequence']

    folderNames = getDirList(rootDirectory)
    folderPaths = [rootDirectory + '/' + _ for _ in folderNames]

    drugDictList = getFileList(drugDictPath)
    drugDictPathList = [drugDictPath + '/' + x for x in drugDictList if x[0]!='.']
    drugDictNameList = [x.split('_')[0].replace(' ','').lower() for x in drugDictList if x[0]!='.']

    proteinDictList = getFileList(proteinDictPath)
    proteinDictPathList = [proteinDictPath + '/' + x for x in proteinDictList if x[0]!='.']
    proteinDictNameList = [x.split('_')[0].replace(' ','').lower() for x in proteinDictList if x[0]!='.']

    for i in range(len(folderPaths)):
        filelist = getFileList(folderPaths[i])
        filePathList = [folderPaths[i] + '/' + x for x in filelist if x[0]!='.']

        drugIndex = drugDictNameList.index(folderNames[i].replace(' ','').lower())
        drugDictPath = drugDictPathList[drugIndex]

        proteinIndex = proteinDictNameList.index(folderNames[i].replace(' ','').lower())
        proteinDictPath = proteinDictPathList[proteinIndex]

        outFileDir = outPath+'/'+folderNames[i]+'_p'
        if os.path.exists(outFileDir):
            files = getFileList(outFileDir)
            if files:
                for file in files:
                    os.remove(outFileDir+'/'+file)
            os.removedirs(outFileDir)
        os.makedirs(outFileDir)

        for j in range(len(filelist)):
            pairPath = filePathList[j]
            outFilePath = outFileDir+'/'+filelist[j]

            processOneFile(columns,pairPath,drugDictPath,proteinDictPath,outFilePath)