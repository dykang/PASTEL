"""
Two stages:
 - per single style, per tuple of all styles

Features :
DONE
 - lexical: n-grams,num-entities, stopwords
 - syntax: length, POS tags of N-grams,
 - deep: glove-avging,
 - noise: oov
 - semantic: sentiment
TBD
- lexical:  tfidf,
- syntax:  shallow parse
- deep: char/word LSTM,
- semantic: sarcam, politeness, metaphor,
- noise: mis-spelling
- discourse: coreference,

Level: given a story = (sent1, ...sent5),
 - level=0 sentence feature outputs [f1, ...f5]
 - level=1 story feature outputs [f]

NOTE:
  python -m spacy download en
  python -m spacy download en_core_web_sm
"""

import os,sys,operator
import numpy as np
from collections import defaultdict
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import spacy

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from reader import load_main_annotations
from utils import import_embeddings

reload(sys)
sys.setdefaultencoding('utf-8')

###########################################################################
def read_features(ifile):
  idToFeature = dict()
  with open(ifile,"rb") as ifh:
    for l in ifh:
      e = l.rstrip().decode("utf-8").split("\t")
      if len(e) == 2:
        idToFeature[e[1]] = e[0]
  return idToFeature

def save_features_to_file(idToFeature, feature_output_file):
  with open(feature_output_file, 'wb') as ofh:
    sorted_items = sorted(idToFeature.items(), key=operator.itemgetter(1))
    for i in sorted_items:
      str = "{}\t{}\n".format(i[1],i[0]).encode("utf-8")
      ofh.write(str)

def get_feature_id(feature):
  if feature in idToFeature:
    return idToFeature[feature]
  else:
    return None

def addFeatureToDict(fname):
  id = len(idToFeature)
  idToFeature[fname] = id


def get_lexical_features(persona_single_dic, ngram_size, max_features, is_lower=False):
  sents = [stories for s,v_dic in persona_single_dic.items() for v, stories in v_dic.items()  ]
  sents =[s for story_style in sents for sent in story_style for s in sent]
  print 'total sentences:',len(sents)
  vect = CountVectorizer(
      ngram_range=(1,ngram_size),
      tokenizer=tokenizer,
      stop_words = 'english',
      lowercase=is_lower,
      max_features = max_features)
  vect.fit(sents)
  ngram_vocab = vect.vocabulary_
  vocab = {k:v for k,v in ngram_vocab.items() if len(k.split(' '))==1}
  print 'ngram Vocab size:', len(ngram_vocab)
  print 'unigram Vocab size:', len(vocab)
  return vect,ngram_vocab,vocab



def add_features(ngram_vocab, emb):
  id2ngram_vocab = {i:w for w,i in ngram_vocab.items()}
  for fid in sorted(id2ngram_vocab.keys()):
    ngram = id2ngram_vocab[fid]
    #import pdb; pdb.set_trace()
    addFeatureToDict(ngram) #'ngram_'+
  for f in range(300):
    addFeatureToDict('emb'+str(f))
  for f in ['f_num_entity','f_num_stopwords', 'f_sent_lens','f_pos_NUM','f_pos_ADP','f_pos_NOUN','f_pos_VERB','f_pos_PROPN','f_pos_ADJ','f_pos_ADV','f_pos_INTJ','f_pos_SYM','f_oov','f_sentiment']:
    addFeatureToDict(f)
  save_features_to_file(idToFeature, feature_output_file)






tokenizer = word_tokenize #TreebankWordTokenizer().tokenize
nlp = spacy.load('en_core_web_sm')

# order matters for svmlight feature files
feature_names = ['f_ngram','f_emb_avg','f_num_entity','f_num_stopwords','f_sent_lens','f_pos_NUM','f_pos_ADP','f_pos_NOUN','f_pos_VERB','f_pos_PROPN','f_pos_ADJ','f_pos_ADV','f_pos_INTJ','f_pos_SYM','f_oov','f_sentiment']
print 'Total number of types of features: ',len(feature_names)
idToFeature = None
feature_output_file = '../../exp/model/' + 'features.dat'
if os.path.isfile(feature_output_file):
  idToFeature = read_features(feature_output_file)
else:
  print 'Loading vector file from scratch..'
idToFeature = dict()






###########################################################################
# main
###########################################################################

def main(args, max_sent=5, ngram_size=3, model_dir = '../../exp/model/', is_lower=True):
  if len(args) != 6:
    print 'USAGE:',args[0],'TBD'

  project = args[1]
  model_dir = args[2]
  max_features = int(args[3]) # maximum number of ngram features (not vocab size)
  level = int(args[4])
  label_type = args[5]

  # (1) load all main annotations
  annots = load_main_annotations(
        dir_mainExp = '../../AMT/AMTResults/mainExp/train/batches_StoryPlusKeywords2/',  max_sent=max_sent)

  # (2) extract per-persona sentences
  persona_single_dic, persona_multi_dic = extract_per_persona_dic(annots) #, limit=500)

  return annots,persona_single_dic,persona_multi_dic


def extract_feature(stories, vect, ngram_vocab, emb,  level, is_lower=False, max_sent=5):
  if not (level == 0 or level==1):
    print 'Wrong level',level
    sys.exit(1)

  cnt_skipped=0
  features = []
  stories_origin = []
  for sid, story in enumerate(stories):
    f_story = {}
    story_tks = [tokenizer(sent.lower()) if is_lower else tokenizer(sent) for sent in story]
    story_nlp = [nlp(sent.decode('utf-8')) for sent in story]

    # lexical
    f_ngram = np.array(vect.transform(story).todense()) #.flatten()
    f_num_entity = [len(sent_nlp.ents) for sent_nlp in story_nlp]
    #TODO extend this to all types of Named Entity such as POS
    f_num_stopwords = [np.sum([1 for w in sent_nlp if w.is_stop]) for sent_nlp in story_nlp]
    # syntax
    f_sent_lens = [len(sent_tks) for sent_tks in story_tks]
    f_pos_NUM = [np.sum([1 for w in sent_nlp if w.pos_ == 'NUM']) for sent_nlp in story_nlp ]
    f_pos_ADP = [np.sum([1 for w in sent_nlp if w.pos_ == 'ADP']) for sent_nlp in story_nlp ]
    f_pos_NOUN = [np.sum([1 for w in sent_nlp if w.pos_ == 'NOUN']) for sent_nlp in story_nlp ]
    f_pos_VERB = [np.sum([1 for w in sent_nlp if w.pos_ == 'VERB']) for sent_nlp in story_nlp ]
    f_pos_PROPN = [np.sum([1 for w in sent_nlp if w.pos_ == 'PROPN']) for sent_nlp in story_nlp ]
    f_pos_ADJ = [np.sum([1 for w in sent_nlp if w.pos_ == 'ADJ']) for sent_nlp in story_nlp ]
    f_pos_ADV = [np.sum([1 for w in sent_nlp if w.pos_ == 'ADV']) for sent_nlp in story_nlp ]
    f_pos_INTJ = [np.sum([1 for w in sent_nlp if w.pos_ == 'INTJ']) for sent_nlp in story_nlp ]
    f_pos_SYM = [np.sum([1 for w in sent_nlp if w.pos_ == 'SYM']) for sent_nlp in story_nlp ]
    # deep
    f_emb_avg = [ np.average([emb[w]  for w in sent if w in emb],axis=0) for sent in story_tks]
    # oov
    f_oov = [np.sum([1 for w in sent_nlp if w.is_oov]) for sent_nlp in story_nlp ]
    # semantic
    f_sentiment = [np.average([w.sentiment for w in sent_nlp]) for sent_nlp in story_nlp ]


    #TODO more aggresivie filtering
    #TODO no images
    try:
      if not all([f.shape[0]==300 for f in f_emb_avg]):
        cnt_skipped += 1
        #print '[EMPTY AVG-EMB] [%d skipped / %d]'%(cnt_skipped,sid)
        continue

      # merge features
      for f_name in feature_names:
        f_value = locals()[f_name]
        f_value = np.array(f_value)
        f_value = np.reshape(f_value, (f_value.shape[0],-1))
        #print ' >',f_name, f_value.shape

        if level==0:
          f_story[f_name] = f_value
        else:
          assert len(f_value) == 5
          f_story[f_name] = np.average(f_value,axis=0)
    except Exception as e:
      cnt_skipped += 1
      #print '[EMPTY AVG-EMB] [%d skipped / %d]'%(cnt_skipped,sid)
      continue

    # print dimension of each feature
    #print ' ' .join(['[%s:%d]'%(k,len(f)) for k,f in f_story.items()])
    features.append(f_story)
    stories_origin.append(story)

  if cnt_skipped >0:
    print ' -Skipped examples %d out of %d'%(cnt_skipped,len(features))
  return features,stories_origin




def extract_features_single_persona(dic, vect, ngram_vocab, emb, model_dir, level=0, is_lower=False, max_sent=5):
  id2ngram_vocab = {i:w for w,i in ngram_vocab.items()}

  label_type = 'single'
  print 'Extracting features from single person (level %d)'%(level)
  for style, value_dic in dic.items():
    fout_model_file = open(
        model_dir + 'svmlite.%s_%s_%d_%d'%(style,label_type,level,len(ngram_vocab)),'w')

    print '======================================='
    print 'STYLE: %s, Value Types: %d' % (style,len(value_dic))
    print 'Value Dist: ',' '.join(['%s:%d'%(k,len(v)) for k,v in value_dic.items()])
    print '======================================='

    value_index_dic = {}
    with open(model_dir+'labels.%s_%s_%d_%d.txt'%(style,label_type,level,len(ngram_vocab)),'w') as fout:
      for vidx,v in enumerate(value_dic.keys()):
        fout.write('%s\t%d\n'%(v,len(value_dic[v])))
        value_index_dic[v] = vidx

    for value,stories in value_dic.items():
      features,stories = extract_feature(
          stories, vect, ngram_vocab, emb, level,is_lower=is_lower, max_sent=max_sent)
      print value, len(features),' samples'
      for feature,story in zip(features,stories):

        if level == 0: # sentence-level
          for sid in range(max_sent):
            fout_model_file.write(str(value_index_dic[value]) + ' ')
            for f_name in feature_names:
              f_value = feature[f_name][sid]
              if f_name == 'f_ngram':
                for f_id, f_v in enumerate(f_value):
                  if f_v == 0 : continue
                  f = id2ngram_vocab[f_id]
                  fid = get_feature_id(f)
                  fout_model_file.write('%s:%s '%(str(fid),f_v))
              elif f_name == 'f_emb_avg':
                for f_id, f_v in enumerate(f_value):
                  if f_v == 0 : continue
                  fid = get_feature_id('emb'+str(f_id))
                  fout_model_file.write('%s:%s '%(str(fid),f_v))
              else:
                fid = get_feature_id(f_name)
                f_value = f_value[0] # every feature values are [x]
                fout_model_file.write('%s:%s '%(str(fid),str(f_value)))
            fout_model_file.write('\n')

        elif level == 1: # story-level
          fout_model_file.write(str(value_index_dic[value]) + ' ')
          for f_name in feature_names:
            f_value = feature[f_name]
            if f_name == 'f_ngram':
              for f_id, f_v in enumerate(f_value):
                if f_v == 0 : continue
                f = id2ngram_vocab[f_id]
                fid = get_feature_id(f)
                fout_model_file.write('%s:%s '%(str(fid),f_v))
            elif f_name == 'f_emb_avg':
              for f_id, f_v in enumerate(f_value):
                if f_v == 0 : continue
                fid = get_feature_id('emb'+str(f_id))
                fout_model_file.write('%s:%s '%(str(fid),f_v))
            else:
              fid = get_feature_id(f_name)
              f_value = f_value[0] # every feature values are [x]
              fout_model_file.write('%s:%s '%(str(fid),str(f_value)))
          fout_model_file.write('\n')

        else:
          print 'Wrong level:',level
          sys.exit(1)

    print 'Done with %s\n'%(style)
    fout_model_file.close()
  print 'Done\n'
  return


# st=(female, left, morning) vs (male, left, morning)

def extract_features_multi_persona(
  dic, vect, ngram_vocab, emb, model_dir, level=0, is_lower=False, max_sent=5):

  id2ngram_vocab = {i:w for w,i in ngram_vocab.items()}
  label_type = 'multi'
  print 'Extracting features from multi persona (level %d)'%(level)
  for style, value_dic in dic.items():
    fout_model_file = open(
        model_dir + 'svmlite.%s_%s_%d_%d'%(style,label_type,level,len(ngram_vocab)),'w')

    print '======================================='
    print 'STYLE: %s, Value Types: %d' % (style,len(value_dic))
    print 'Value Dist: ',' '.join(['%s:%d'%(k,len(v)) for k,v in value_dic.items()])
    print '======================================='

    value_index_dic = {}
    with open(model_dir+'labels.%s_%s_%d_%d.txt'%(style,label_type,level,len(ngram_vocab)),'w') as fout:
      for vidx,v in enumerate(value_dic.keys()):
        fout.write('%s\t%d\n'%(v,len(value_dic[v])))
        value_index_dic[v] = vidx

    for value,stories in value_dic.items():
      features,stories = extract_feature(
          stories, vect, ngram_vocab, emb, level,is_lower=is_lower, max_sent=max_sent)
      print value, len(features),' samples'
      for feature,story in zip(features,stories):

        if level == 0: # sentence-level
          for sid in range(max_sent):
            fout_model_file.write(str(value_index_dic[value]) + ' ')
            for f_name in feature_names:
              f_value = feature[f_name][sid]
              if f_name == 'f_ngram':
                for f_id, f_v in enumerate(f_value):
                  if f_v == 0 : continue
                  f = id2ngram_vocab[f_id]
                  fid = get_feature_id(f)
                  fout_model_file.write('%s:%s '%(str(fid),f_v))
              elif f_name == 'f_emb_avg':
                for f_id, f_v in enumerate(f_value):
                  if f_v == 0 : continue
                  fid = get_feature_id('emb'+str(f_id))
                  fout_model_file.write('%s:%s '%(str(fid),f_v))
              else:
                fid = get_feature_id(f_name)
                f_value = f_value[0] # every feature values are [x]
                fout_model_file.write('%s:%s '%(str(fid),str(f_value)))
            fout_model_file.write('\n')

        elif level == 1: # story-level
          fout_model_file.write(str(value_index_dic[value]) + ' ')
          for f_name in feature_names:
            f_value = feature[f_name]
            if f_name == 'f_ngram':
              for f_id, f_v in enumerate(f_value):
                if f_v == 0 : continue
                f = id2ngram_vocab[f_id]
                fid = get_feature_id(f)
                fout_model_file.write('%s:%s '%(str(fid),f_v))
            elif f_name == 'f_emb_avg':
              for f_id, f_v in enumerate(f_value):
                if f_v == 0 : continue
                fid = get_feature_id('emb'+str(f_id))
                fout_model_file.write('%s:%s '%(str(fid),f_v))
            else:
              fid = get_feature_id(f_name)
              f_value = f_value[0] # every feature values are [x]
              fout_model_file.write('%s:%s '%(str(fid),str(f_value)))
          fout_model_file.write('\n')

        else:
          print 'Wrong level:',level
          sys.exit(1)

    print 'Done with %s\n'%(style)
    fout_model_file.close()
  print 'Done\n'
  return











def extract_per_persona_dic(annots, verbose=False, limit=False):
  print 'Extracting persona dictionaries...'
  # persona dictionaries
  persona_single_story_dic = defaultdict(lambda: defaultdict(list)) #defaultdict(list)
  persona_all_story_dic = defaultdict(list)


  for aid, annot in enumerate(annots):
    if limit and aid > limit: break


    if verbose:
      print 'StoryID: %s, persona num: %d'%(annot,len(annots[annot]))
    for pid,persona in enumerate(annots[annot]):
      if verbose:
        print 'PID: %d'%(pid)
      #print persona.keys()
      style_keys = persona.keys()
      style_keys.sort()

      styles = {}
      sents = {}
      for style in style_keys:
        value = persona[style]
        if style.startswith('Answer.story'):
          if verbose:
            print 'Story:',style,value
          sents[int(style.replace('Answer.story',''))] = value
        elif style.startswith('Answer.'):
          if verbose:
            print 'Persona:',style, value
          styles[style.replace('Answer.','')] = value

      sents = [sents[i] for i in range(1,6,1)]

      # merge them into persona dictionaries
      for s,v in styles.items():
        persona_single_story_dic[s][v] += [sents]

      # NOTE check the order of styles though
      style_value_tuple = tuple([v for s,v in styles.items()])
      assert len(style_value_tuple) == 8
      persona_all_story_dic[style_value_tuple] += [sents]

  print 'Persona Single Story Dic',persona_single_story_dic.keys()
  print 'Persona Value Combinations',' '.join(['%s:%d'%(k,len(v)) for k,v in persona_single_story_dic.items()])

  for style, value_dic in persona_single_story_dic.items():
    print '- STYLE: %s, Value Types: %d' % (style,len(value_dic))
    print ' * Value Dist: ',' '.join(['%s:%d'%(k,len(v)) for k,v in value_dic.items()])

  print 'Persona All Story Dic',len(persona_all_story_dic)
  print 'Done\n'
  return persona_single_story_dic, persona_all_story_dic


if __name__ == '__main__': 
  annots,persona_single_dic,persona_multi_dic=main(sys.argv)
  print persona_single_dic
  import pickle
  pickle.dump(persona_single_dic,open("persona_single_dic.p","wb"))
  pickle.dump(persona_multi_dic,open("persona_multi_dic.p","wb"))
  pickle.dump(annots,open("annots.p","wb"))
