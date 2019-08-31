import os
import sys
import operator
import copy
import json
import pickle
import random
from nltk.tokenize import word_tokenize

import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import spacy

from reader import load_main_annotations

STYLE_ORDER = ['gender', 'country', 'age', 'ethnic', 'education', 'politics', 'tod']

def wrapInTags(s):
    return '<'+s+'>'

def convertToAscii(s):
    return s
    # FIXEd: no need for python3
    # return s.encode('ascii','ignore')

def tokenizeAndRejoin(s):
    words=word_tokenize(s)
    words=" ".join(words)
    return words

def constructStyleHeader(annotation,prefix="STYLED"):
    if prefix=="STYLED":
        styleKeys=STYLE_ORDER
    else:
        styleKeys=[prefix]

    try:
        styleHeaderStrings=[wrapInTags(convertToAscii(annotation["persona"][styleKey])) for styleKey in styleKeys]
    except Exception as e:
        print (e)
        import pdb; pdb.set_trace()
    styleHeaderString=" ".join(styleHeaderStrings)
    return styleHeaderString

def convertAnnotationToPair(annotation,prefix="STYLED",story=False):
    if not story:
        sourceSentence=convertToAscii(annotation["input.sentences"])
        targetSentence=convertToAscii(annotation["output.sentences"])
    else:
        sourceSentence=convertToAscii(" SENTEND ".join(annotation["input.sentences"]))
        targetSentence=convertToAscii(" SENTEND ".join(annotation["output.sentences"]))

    styleHeader=constructStyleHeader(annotation,prefix=prefix)

    sourceSentence=tokenizeAndRejoin(sourceSentence)
    targetSentence=tokenizeAndRejoin(targetSentence)

    sourceSentence=styleHeader+" "+sourceSentence

    return (sourceSentence,targetSentence)

def filterAway(annotation,sentenceId,minLength=4):
  #If this returns True, that means the sentence in question should be dropped.
  sourceSentence=convertToAscii(annotation["Input.sent_"+str(sentenceId)])
  targetSentence=convertToAscii(annotation["Answer.story"+str(sentenceId)])

  shouldBeFilteredAway=False

  targetWords=targetSentence.split()

  if len(targetWords)<minLength:
    #or alpha or beta or gamma (Add as many conditions to filter away)
    shouldBeFilteredAway=True
  return shouldBeFilteredAway

# sentence-level, only filter sentences
# story-level, filter stories
def filter_annots(annots,level=-1):
  new_annots = {}
  cnt_sent, cnt_story = 0 ,0
  cnt_filtered_sent, cnt_filtered_story = 0, 0
  for storyId in annots:
    new_story_annots = []
    for annotId, elem in enumerate(annots[storyId]):
      story_filtered = False
      cnt_story += 1
      for sentenceId in range(1,6):
        cnt_sent += 1
        if filterAway(annots[storyId][annotId],sentenceId):
          cnt_filtered_sent += 1
          story_filtered = True
      if story_filtered:
        cnt_filtered_story += 1
      else:
        # if level=='story' or level=='sentence':
        new_story_annots.append(elem)
    new_annots[storyId] = new_story_annots
  print ('Story filtered/total: %d/%d'%(cnt_filtered_story,cnt_story))
  print ('Sentence filtered/total: %d/%d'%(cnt_filtered_sent,cnt_sent))
  return new_annots





def constructPairsFromAnnots(trainAnnots,validAnnots,testAnnots,prefix="STYLED",story=False):
    splits={"train": trainAnnots, "valid":validAnnots, "test":testAnnots}
    splitPairs={}
    for splitName,splitAnnots in splits.items():
        splitPairs[splitName]=[] #All pairs in this split
        for annot in splitAnnots:
            sentencePair=convertAnnotationToPair(annot,prefix=prefix,story=story)
            splitPairs[splitName].append(sentencePair)
    return splits,splitPairs

def writeSplitsToFiles(splits,splitPairs,outputDir="./",prefix="STYLED"):
    for splitName in splitPairs.keys():
        #Construct outfilename
        outFileNamePrefix=outputDir+splitName+"."+prefix
        outFileNameSrc=outFileNamePrefix+".src"
        outFileNameTgt=outFileNamePrefix+".tgt"
        #Open file
        outFileSrc=open(outFileNameSrc,"w")
        outFileTgt=open(outFileNameTgt,"w")
        for splitPair in splitPairs[splitName]:
            srcSentence=splitPair[0]
            tgtSentence=splitPair[1]
            outFileSrc.write(srcSentence+"\n")
            outFileTgt.write(tgtSentence+"\n")
        outFileSrc.close()
        outFileTgt.close()





def convert_to_instance(annots,out_dir='./',verbose=False):
  print ('Saving annotations to story json files...')
  # persona dictionaries
  persona_single_story_dic = defaultdict(lambda: defaultdict(list)) #defaultdict(list)

  new_stories = []
  new_sentences = []
  for aid, annot in annots.items():
    if verbose:
      print ('StoryID: %s, persona num: %d'%(annot,len(annots[annot])))

    for pid,story in enumerate(annot):
      new_story = {}
      #new_story['worker_id'] = story['WorkerId']
      # import pdb; pdb.set_trace()
      new_story['annotation_id'] = story['Input.exampleId']
      new_story['per_annotation_id'] = story['Input.exampleId']
      new_story['id'] = '%s_%s'%(new_story['annotation_id'],pid)

      persona = {}
      persona['country'] = story['Answer.Country']
      persona['politics'] = story['Answer.Politics']
      persona['tod'] = story['Answer.TOD']
      persona['age'] = story['Answer.age']
      persona['education'] = story['Answer.edu']
      persona['ethnic'] = story['Answer.ethnic']
      persona['gender'] = story['Answer.gender']
      #new_story['persona.ease'] = story['Answer.ease']
      new_story['persona'] = persona


      input_keywords = [k.strip() for k in story['Input.tags'].split('-->')]
      input_images,input_sents,output_sents = [],[],[]
      for l in range(1,6,1):
        input_images.append(story['Input.img_%d'%(l)])
        input_sents.append(story['Input.sent_%d'%(l)])
        output_sents.append(story['Answer.story%d'%(l)])


      for idx,(in_k,in_img,in_snt,out_snt) in enumerate(zip(
          input_keywords,input_images,input_sents,output_sents)):
        new_sent = copy.deepcopy(new_story)
        new_sent['input.keywords'] = in_k
        new_sent['input.images'] = in_img
        new_sent['input.sentences'] = in_snt
        new_sent['output.sentences'] = out_snt
        new_sent['id'] = new_sent['id']+'_%d'%(idx)
        new_sentences.append(new_sent)

      new_story['input.keywords'] = input_keywords
      new_story['input.images'] = input_images
      new_story['input.sentences'] = input_sents
      new_story['output.sentences'] = output_sents
      new_stories.append(new_story)

  print('Total stories and sentences: ',len(new_stories),len(new_sentences))
  random.shuffle(new_stories)
  random.shuffle(new_sentences)

  return new_stories, new_sentences



def split_by(stories, ratio=0.8):
    allKeys=[annot['id'] for annot in stories]
    random.shuffle(allKeys)
    allPoints=len(allKeys)
    trainPoints=int(ratio*allPoints)
    validPoints=int((allPoints - trainPoints)/2)
    testPoints=allPoints-trainPoints-validPoints

    trainKeys=allKeys[:trainPoints]
    validKeys=allKeys[trainPoints:trainPoints+validPoints]
    testKeys=allKeys[trainPoints+validPoints:]

    print ("Number of Total Stories:",len(allKeys))
    print ("Number of Train Keys:",len(trainKeys))
    print ("Number of Valid Keys:",len(validKeys))
    print ("Number of Test Keys:",len(testKeys))

    train = [annot for annot in stories if annot['id'] in trainKeys]
    valid = [annot for annot in stories if annot['id'] in validKeys]
    test = [annot for annot in stories if annot['id'] in testKeys]
    return train,valid,test


def save_to_json(objs,out_dir='./',sub_dir='./'):
  # save to json

  directory = os.path.join(out_dir,sub_dir)
  if not os.path.exists(directory):
    os.makedirs(directory)

  for story in objs:
    with open(os.path.join(directory,'%s.json'%(story['id'])),'w') as fout:
      json.dump(story,fout)

def main(args):
    random.seed(786)
    project = args[1]
    data_dir = args[2]
    model_dir = args[3]
    out_dir = args[4]
    transfer_dir = args[5]

    # (1) load all main annotations
    annots = load_main_annotations(dir_mainExp = data_dir) #,  max_sent=max_sent)

    # (2) adding filterAway (from Varun) here
    annots = filter_annots(annots) #, level=level)

    # (3) convert annotation to new story format
    stories,sentences = convert_to_instance(annots)

    # (4) split by train/valid/test
    stories_train, stories_valid, stories_test = split_by(stories, 0.8)
    sents_train, sents_valid, sents_test = split_by(sentences, 0.8)

    # (5) save to json
    save_to_json(stories_train,out_dir=out_dir,sub_dir='stories/train')
    save_to_json(stories_valid,out_dir=out_dir,sub_dir='stories/valid')
    save_to_json(stories_test,out_dir=out_dir,sub_dir='stories/test')
    save_to_json(sents_train,out_dir=out_dir,sub_dir='sentences/train')
    save_to_json(sents_valid,out_dir=out_dir,sub_dir='sentences/valid')
    save_to_json(sents_test,out_dir=out_dir,sub_dir='sentences/test')


    # #TODO Varun, you should change following code accordingly with the new update
    prefixes=["STYLED",]+STYLE_ORDER
    #Outputting from sentences/
    for prefix in prefixes:
        splits,splitPairs=constructPairsFromAnnots(sents_train,sents_valid,sents_test,prefix=prefix)
        writeSplitsToFiles(splits,splitPairs,outputDir=transfer_dir,prefix=prefix)
    #Outputting from stories/
    for prefix in prefixes:
        splits,splitPairs=constructPairsFromAnnots(stories_train,stories_valid,stories_test,prefix=prefix,story=True)
        writeSplitsToFiles(splits,splitPairs,outputDir=transfer_dir,prefix=prefix+"-story")


if __name__ == "__main__": main(sys.argv)

