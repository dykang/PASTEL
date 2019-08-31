import os
import sys
import gensim
import numpy as np

STYLE_ORDER = ['gender', 'country', 'age', 'ethnic', 'education', 'politics', 'tod']

def import_embeddings(filename, vocab=False, project=False):
    # import pdb; pdb.set_trace()
    if vocab is False:
        print('Loading gensim embedding from ',filename)
        glove_embedding = gensim.models.KeyedVectors.load_word2vec_format(
                filename, binary=True)
        return glove_embedding
    else:
        filename_vocab = filename+'.%s.%d' % (project,len(vocab))
        if not os.path.isfile(filename_vocab):
            fout = open(filename_vocab, 'w')
            for line in open(filename, 'r'):
                splitLine = line.strip().split(' ')
                word = splitLine[0]
                if word not in vocab:
                    continue
                fout.write('%s\n' % (line.strip()))
            fout.close()

        model = {}
        try:
            for line in open(filename_vocab, 'r'):
                splitLine = line.strip().split(' ')
                word = splitLine[0]
                if word not in vocab:
                    continue
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
        except Exception as e:
            print (e)
            import pdb
            pdb.set_trace()
        return model


