import xlrd
import os
import gensim


def getLines(dir):
    lines = []
    for file in [_ for _ in os.listdir(dir) if _[0] != '.']:
        print(file)
        workbook = xlrd.open_workbook(os.path.join(dir, file))
        sheet = workbook.sheet_by_index(0)
        linesOfFile = sheet.col_values(1)[1:]
        linesOfFile = [x.encode('utf-8') for x in linesOfFile]
        lines.extend(linesOfFile)
    return lines


def writeLinesToTxt(lines, filepath, filename):
    with open(os.path.join(os.path.join(filepath, '..'), filename), 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        i = 0
        for line in open(self.dirname, 'r'):
            i += 1
            print(i)
            sentence = ' '.join(line)
            print(sentence)
            yield sentence


def my_rule(word, count, min_count):
    if word == ' ':
        return gensim.utils.RULE_DISCARD
    elif word == '\n':
        return gensim.utils.RULE_DISCARD
    else:
        return gensim.utils.RULE_KEEP


drugDir = '/Users/stern/Desktop/Dataset/LargeScale/DictFile/drugLines.txt'
proteinDir = '/Users/stern/Desktop/Dataset/LargeScale/DictFile/proteinLines.txt'

para = 'drug'
test = 1
if para == 'drug':
    sentences = MySentences(drugDir)  # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, size=160, workers=5, trim_rule=my_rule)
    model1 = gensim.models.Word2Vec(sentences, size=160, workers=5, trim_rule=my_rule)
    model.wv.save_word2vec_format(os.path.join(os.path.split(drugDir)[0], 'drug'))
    if test:
        print(model.wv.most_similar('C'))
        model1.wv.load_word2vec_format(
            '/Users/stern/PycharmProjects/Stern/DeepDTA-master/data/vec/drug.chembl.canon.l1.ws20.txt')
        print(model1.wv.most_similar('C'))
else:
    sentences = MySentences(proteinDir)  # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, size=160, workers=5, trim_rule=my_rule)
    model.wv.save_word2vec_format(os.path.join(os.path.split(drugDir)[0], 'protein'))
    if test:
        print(model.wv.most_similar('C'))
