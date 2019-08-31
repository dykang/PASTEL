import numpy as np


def loadEmbedsAsNumpyObj(path,wids,embeddingMatrix):

    embeddingSize=embeddingMatrix.shape[1]
    for line in open(path):
        words=line.strip(' ').split()
        key=words[0]
        try:
            if key in wids:
                values=[float(x) for x in words[1:]][:embeddingSize]
                embeddingMatrix[wids[key]]=np.array(values)
        except Exception as e:
            # print(e)
            continue

    return embeddingMatrix
