import sys
import edit_distance
import numpy as np
import pickle

def findEditDistance(ref,hyp,characterLevel=False,prune=True):
    if characterLevel:
        ref=" ".join(ref)
        hyp=" ".join(hyp)
        ref=[c for c in ref]
        hyp=[c for c in hyp]
    
    if prune:
        refSet=set(ref)
        hypSet=set(hyp)

        if refSet.intersection(hypSet)<=2:
            return 5000.0

    sm=edit_distance.SequenceMatcher(a=ref,b=hyp)
    return sm.distance()

#Tbh this is a bit of a misnomer. glove/embeddingDict.p has been created from the fastText simpleEnglish, i.e wiki.simple.vec embeddings.
gloveDict=pickle.load(open("glove/embeddingDict.p","rb"))

def getVec(word):
    if word in gloveDict:
        return gloveDict[word]
    else:
        return np.zeros((300,))

def getCBOWVec(wordList):
    CBOWVec=np.zeros((300,))
    for word in wordList:
        CBOWVec+=getVec(word)
    CBOWVec+=getVec("</s>")
    CBOWVec=CBOWVec/(len(wordList)+1.0)
    return CBOWVec

def findCBOWDistance(ref,hyp):
    refVec=getCBOWVec(ref)
    hypVec=getCBOWVec(hyp)
    dotProduct=np.dot(refVec,hypVec)
    magA=np.dot(hypVec,hypVec)
    magB=np.dot(refVec,refVec)
    magA=magA**0.5
    magB=magB**0.5
    dotProduct=dotProduct/((magA+1e-5)*(magB+1e-5))
    return 1.0/dotProduct

headerLength=int(sys.argv[2])
prefix=sys.argv[1]

srcFile=open("data/train."+prefix+".src")
tgtFile=open("data/train."+prefix+".tgt")

headerToSentenceMapper={}

allLines=zip(srcFile.readlines(),tgtFile.readlines())

for line in allLines:
    srcLineWords=line[0].split()
    tgtLineWords=line[1].split()

    header=" ".join(srcLineWords[:headerLength])
    srcLineWords=srcLineWords[headerLength:]

    if header not in headerToSentenceMapper:
        headerToSentenceMapper[header]=[]

    headerToSentenceMapper[header].append((srcLineWords,tgtLineWords))



srcFile=open("data/test."+prefix+".src")
tgtFile=open("data/test."+prefix+".src.delete","w")

threshold=0.9

for lineIndex,line in enumerate(srcFile.readlines()):
    if lineIndex%100==0:
        print "Line Index:",lineIndex
    srcLineWords=line.split()

    header=" ".join(srcLineWords[:headerLength])
    srcLineWords=srcLineWords[headerLength:]

    if header not in headerToSentenceMapper:
        outputWords=srcLineWords
    else:
        candidateSentences=headerToSentenceMapper[header]
        minEditDistance=1000
        minEditDistanceIndex=0
        for candidateIndex,candidateSentence in enumerate(candidateSentences):
            ref=srcLineWords
            hyp=candidateSentence[0]
            pruneCandidateEditDistance=findEditDistance(ref,hyp)
            if pruneCandidateEditDistance>20:
                continue
            candidateDistance=findCBOWDistance(ref,hyp)
            
            if candidateDistance<minEditDistance:
                minEditDistance=candidateDistance
                minEditDistanceIndex=candidateIndex
        outputWords=candidateSentences[minEditDistanceIndex][1]

    outputLine=" ".join(outputWords)+"\n"
    tgtFile.write(outputLine)

tgtFile.close()
