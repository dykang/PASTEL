import numpy as np
import pickle 

fp=open("wiki.simple.vec")

embeddingDict={}
for line in fp.readlines():
    words=line.split()
    key=words[0]
    valueVector=np.asarray([float(x) for x in words[1:]])
    embeddingDict[key]=valueVector

pickle.dump(embeddingDict,open("embeddingDict.p","wb"))





