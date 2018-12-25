from pychem import getmol
import pandas as pd

def getDrugEntry(file):
    """
    Get Drug Entry from downloaded FTP file drug;
    :return: list of drug entrys(which means id);
    """
    entrys = []
    with open(file,'r') as f:
        line = f.readline()
        count = 0
        while line and line.strip():
            if line.split()[0] == 'ENTRY':
                entrys.append(line.split()[1])
                count +=1
            line = f.readline()
        # print("the number of the entry in Database:"+count)
    return entrys


def writeToFile(entrys,outPath):
    """
    Write the smiles got from entry to external file
    :param entrys:
    :return:
    """
    df_kegg_sm = pd.DataFrame(columns=['kegg_id','smiles'])
    i = -1
    for entry in entrys:
        i += 1
        try:
            smi = getmol.GetMolFromKegg(entry)
        except:
            smi = 'None'

        print '======================='
        print entry
        print i
        print smi
        df_kegg_sm.loc[i,'kegg_id'] = entry
        df_kegg_sm.loc[i,'smiles'] = smi
    df_kegg_sm.to_csv(outPath, encoding='utf-8', index=False)


if __name__ == '__main__':

    file = '/Users/stern/Desktop/Dataset/KEGG/drug'
    outPath = "/Users/stern/Desktop/Dataset/KEGG/kegg_smiles.csv"
    entrys = getDrugEntry(file)
    writeToFile(entrys,outPath)