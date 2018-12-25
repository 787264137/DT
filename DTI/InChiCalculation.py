import pybel
import pandas as pd


def getSmiles(file, smileColumn):
    smiles = []
    with open(file, 'r') as f:
        line = f.readline()
        while line and line.strip():
            smiles.append(line.split(',')[smileColumn - 1])
            line = f.readline()
    return smiles


def writeToFile(smiles, outPath):
    """
    Write the smiles got from entry to external file
    :param entrys:
    :return:
    """
    df_kegg_sm = pd.DataFrame(columns=['smiles', 'InChi'])
    i = -1
    j = 0
    for smile in smiles:
        i += 1
        try:
            mol = pybel.readstring('smi', smile)
            inChi = mol.write('inchi')
        except:
            inChi = 'None'
            j += 1

        print '======================='
        print smile
        print i
        print inChi
        df_kegg_sm.loc[i, 'smiles'] = smile
        df_kegg_sm.loc[i, 'InChi'] = inChi
    df_kegg_sm.to_csv(outPath, encoding='utf-8', index=False)
    print 'SUM:%d' % (i + 1)
    print 'None:%d' % j


if __name__ == '__main__':
    """KEGG
        SUM:1279
        None:52
    """
    # fileKEGG = '/Users/stern/Desktop/Dataset/KEGG/kegg_smiles.csv'
    # outPathKEGG = "/Users/stern/Desktop/Dataset/KEGG/kegg_InChi.csv"
    # smilesKEGG = getSmiles(fileKEGG, 2)
    # writeToFile(smilesKEGG, outPathKEGG)
    """DrugBank
        SUM:7370
        None:1548
    """
    fileDrugBank = '/Users/stern/Desktop/Dataset/DrugBank/drugbank_smiles.csv'
    outPathDrugBank = '/Users/stern/Desktop/Dataset/DrugBank/drugbank_InChi.csv'
    smilesDrugBank = getSmiles(fileDrugBank, 4)
    writeToFile(smilesDrugBank, outPathDrugBank)
