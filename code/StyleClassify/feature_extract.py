import os
import sys
import glob
import json
import operator
import numpy as np
from collections import defaultdict
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from utils import import_embeddings,STYLE_ORDER

tokenizer = word_tokenize #TreebankWordTokenizer().tokenize
nlp = spacy.load('en_core_web_sm')

filter_dic={}
filter_dic[0] = ['Female','Male']
filter_dic[2] = ['18-24','35-44']
filter_dic[4] = ['Bachelor','NoDegree']
filter_dic[5] = ['LeftWing','RightWing']


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





def get_lexical_features(data, level, ngram_size, max_features, is_lower=False):
    print ('Extracting Ngram and embedding of uni-gram vocabs..')
    sents = [d['output.sentences'] for d in data]
    if level == 'stories':
        sents = [o for s in sents for o in s]
    print ('total sentences:',len(sents))
    vect = CountVectorizer(
            ngram_range=(1,ngram_size),
            tokenizer=tokenizer,
            stop_words = 'english',
            lowercase=is_lower,
            max_features = max_features)
    vect.fit(sents)
    ngram_vocab = vect.vocabulary_
    vocab = {k:v for k,v in ngram_vocab.items() if len(k.split(' '))==1}
    print ('ngram Vocab size:', len(ngram_vocab))
    print ('unigram Vocab size:', len(vocab))
    return vect,ngram_vocab,vocab



def add_features(ngram_vocab, emb):
    id2ngram_vocab = {i:w for w,i in ngram_vocab.items()}
    for fid in sorted(id2ngram_vocab.keys()):
        ngram = id2ngram_vocab[fid]
        addFeatureToDict(ngram) #'ngram_'+
    for f in range(300):
        addFeatureToDict('emb'+str(f))
    for f in ['f_num_entity','f_num_stopwords', 'f_sent_lens','f_pos_NUM','f_pos_ADP','f_pos_NOUN','f_pos_VERB','f_pos_PROPN','f_pos_ADJ','f_pos_ADV','f_pos_INTJ','f_pos_SYM','f_oov','f_sentiment']:
        addFeatureToDict(f)
    save_features_to_file(idToFeature, feature_output_file)



# order matters for svmlight feature files
feature_names = ['f_ngram','f_emb_avg','f_num_entity','f_num_stopwords','f_sent_lens','f_pos_NUM','f_pos_ADP','f_pos_NOUN','f_pos_VERB','f_pos_PROPN','f_pos_ADJ','f_pos_ADV','f_pos_INTJ','f_pos_SYM','f_oov','f_sentiment']
print ('Total number of types of features: ',len(feature_names))
idToFeature = None
feature_output_file = '../../exp/model/' + 'features.dat'
if os.path.isfile(feature_output_file):
    idToFeature = read_features(feature_output_file)
else:
    print ('Loading vector file from scratch..')
idToFeature = dict()



def extract_feature_from_sentence(sent,vect,ngram_vocab,emb, is_lower=False):
    f_sent = {}
    sent_tks = tokenizer(sent.lower()) if is_lower else tokenizer(sent)
    sent_nlp = nlp(sent)

    # lexical
    f_ngram = np.array(vect.transform([sent]).todense().tolist()[0]) #.flatten()
    f_num_entity = len(sent_nlp.ents)
    f_num_stopwords = np.sum([1 for w in sent_nlp if w.is_stop])
    # syntax
    f_sent_lens = len(sent_tks)
    f_pos_NUM = np.sum([1 for w in sent_nlp if w.pos_ == 'NUM'])
    f_pos_ADP = np.sum([1 for w in sent_nlp if w.pos_ == 'ADP'])
    f_pos_NOUN = np.sum([1 for w in sent_nlp if w.pos_ == 'NOUN'])
    f_pos_VERB = np.sum([1 for w in sent_nlp if w.pos_ == 'VERB'])
    f_pos_PROPN = np.sum([1 for w in sent_nlp if w.pos_ == 'PROPN'])
    f_pos_ADJ = np.sum([1 for w in sent_nlp if w.pos_ == 'ADJ'])
    f_pos_ADV = np.sum([1 for w in sent_nlp if w.pos_ == 'ADV'])
    f_pos_INTJ = np.sum([1 for w in sent_nlp if w.pos_ == 'INTJ'])
    f_pos_SYM = np.sum([1 for w in sent_nlp if w.pos_ == 'SYM'])
    # deep
    tks_in_emb = [w for w in sent_tks if w in emb]
    if len(tks_in_emb) == 0:
        return None,None
    f_emb_avg = np.average([emb[w] for w in tks_in_emb],axis=0)
    # oov
    f_oov = np.sum([1 for w in sent_nlp if w.is_oov])
    # semantic
    f_sentiment = np.average([w.sentiment for w in sent_nlp])

    # merge features
    for f_name in feature_names:
        f_value = locals()[f_name]
        f_value = np.array(f_value)
        f_sent[f_name] = f_value
    #print (' ' .join(['[%s:%d]'%(k,len(f)) for k,f in f_story.items()]))
    return f_sent, sent


def extract_feature(stories, vect, ngram_vocab, emb,level, is_lower=False):
    if not (level in ['sentences','stories']):
        print ('Wrong level',level)
        sys.exit(1)
    cnt_skipped=0
    features = []
    stories_origin = []
    for sid, obj in enumerate(stories):
        if level == 'sentences':
            f_sent, sent = extract_feature_from_sentence(obj,vect,ngram_vocab,emb,is_lower=is_lower)
        elif level == 'stories':
            f_sent_avg, sent_avg = [], []
            for sent in obj:
                f_s, s = extract_feature_from_sentence(sent,vect,ngram_vocab,emb,is_lower=is_lower)
                if f_s is None or s is None:
                    continue
                f_sent_avg.append(f_s)
                sent_avg.append(s)
            if len(f_sent_avg) ==5:
                f_sent = {}
                for f_name in feature_names:
                    f_sent[f_name] = np.average([f_sent[f_name] for f_sent in f_sent_avg],axis=0)
                sent = sent_avg
            else:
                f_sent = None
        if f_sent is None or sent is None:
            cnt_skipped += 1
            continue
        features.append(f_sent)
        stories_origin.append(sent)

    if cnt_skipped >0:
        print (' -Skipped examples %d out of %d'%(cnt_skipped,len(features)))
    return features,stories_origin




def extract_features_combined_persona(dic, vect, ngram_vocab, emb, model_dir, level=0, is_lower=False):
    id2ngram_vocab = {i:w for w,i in ngram_vocab.items()}

    exp_setting = 'combined'
    if not os.path.exists(os.path.join(model_dir,exp_setting)):
        os.makedirs(os.path.join(model_dir,exp_setting))
    print ('Extracting features from combined person (level %s)'%(level))
    for style, value_dic in dic.items():

        fout_model_file = open(
                model_dir + 'combined/svmlite.%s_%s_%s_%d'%(style,exp_setting,level,len(ngram_vocab)),'w')

        if style not in STYLE_ORDER:
            print ('No style in style order list',style)
            sys.exit(1)

        style_index = STYLE_ORDER.index(style)
        if not style_index in filter_dic:
            print ('Style index not in filder dictionary',style)
            continue
        only_use_values = filter_dic[style_index]

        new_value_dic = {}
        for k,v in value_dic.items():
            if k in only_use_values:
                new_value_dic[k] = v
        value_dic = new_value_dic

        print ('=======================================')
        print ('STYLE: %s, Value Types: %d' % (style,len(value_dic)))
        print ('Value Dist: ',' '.join(['%s:%d'%(k,len(v)) for k,v in value_dic.items()]))
        print ('=======================================')

        value_index_dic = {}
        with open(model_dir+'combined/labels.%s_%s_%s_%d.txt'%(style,exp_setting,level,len(ngram_vocab)),'w') as fout:
            for vidx,v in enumerate(value_dic.keys()):
                fout.write('%s\t%d\n'%(v,len(value_dic[v])))
                value_index_dic[v] = vidx

        for value,stories in value_dic.items():
            features,stories = extract_feature(
                    stories, vect, ngram_vocab, emb, level,is_lower=is_lower)
            print (value, len(features),len(stories),' samples')
            for feature,story in zip(features,stories):
                if level in ['stories','sentences']:
                    # for sid in range(max_sent):
                    fout_model_file.write(str(value_index_dic[value]) + ' ')
                    for f_name in feature_names:
                        f_value = feature[f_name] #[sid]
                        # import pdb; pdb.set_trace()
                        if f_name == 'f_ngram':
                            for f_id, f_v in enumerate(f_value):
                                if f_v == 0 :
                                    continue
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
                            f_value = f_value #[0] # every feature values are [x]
                            fout_model_file.write('%s:%s '%(str(fid),str(f_value)))
                    fout_model_file.write('\n')
                else:
                    print ('Wrong level:',level)
                    sys.exit(1)

        print ('Done with %s\n'%(style))
        fout_model_file.close()
    print ('Done\n')
    return


def extract_features_controlled_persona(
    dic, vect, ngram_vocab, emb, model_dir, level=0, is_lower=False):
    id2ngram_vocab = {i:w for w,i in ngram_vocab.items()}
    exp_setting = 'controlled'
    if not os.path.exists(os.path.join(model_dir,exp_setting)):
        os.makedirs(os.path.join(model_dir,exp_setting))
    print ('Extracting features from controlled persona (level %s)'%(level))

    for styleId in filter_dic.keys():
        if styleId not in filter_dic: continue

        main_style_type = STYLE_ORDER[styleId]

        # print ('Style ID: ', styleId)
        tuple_style_dic = defaultdict(lambda: defaultdict(list))
        for style, value_dic in dic.items():
            # print (style[styleId], style)

            # check other_style in filter_dic
            noValueExist = False
            for sid in filter_dic.keys():
                if style[sid] not in filter_dic[sid]:
                    noValueExist = True
                    break
            if noValueExist: continue

            main_style = style[styleId]
            other_style = tuple([style[i] for i in filter_dic.keys() if i!=styleId])
            # print (main_style, other_style, len(value_dic))

            tuple_style_dic[other_style][main_style] += value_dic


        print ('=======================================')
        print ('STYLE: %s, Value Types: %d, size: %d' % (
            main_style_type,len(tuple_style_dic),len(tuple_style_dic)))
        for k,v in tuple_style_dic.items():
            print (k, len(v), ', '.join(['%s:%d'%(vk,len(vv)) for vk, vv in v.items()]))
        print ('=======================================')

        for other_style_tuple, other_style_dic in tuple_style_dic.items():
            other_style_tuple_str = ','.join(other_style_tuple)
            value_index_dic = {}

            if len(other_style_dic) < 2:
                print ('Skipped',other_style_tuple, len(other_style_dic))
                continue

            with open(model_dir+'controlled/labels.%s_%s_%s_%s_%d.txt'%(
                main_style_type,other_style_tuple_str,exp_setting,level,len(ngram_vocab)),'w') as fout:
                for vidx,v in enumerate(other_style_dic.keys()):
                    fout.write('%s\t%d\n'%(v,len(other_style_dic[v])))
                    value_index_dic[v] = vidx

            with open(model_dir + 'controlled/svmlite.%s_%s_%s_%s_%d'%(
                main_style_type,other_style_tuple_str,exp_setting,level,len(ngram_vocab)),'w') as fout_model_file:
                for value,stories in other_style_dic.items():
                    features,stories = extract_feature(
                            stories, vect, ngram_vocab, emb, level,is_lower=is_lower)
                    # print (value, len(features),' samples')
                    for feature,story in zip(features,stories):
                        if level == 'sentences': # sentence-level
                            #for sid in range(max_sent):
                            fout_model_file.write(str(value_index_dic[value]) + ' ')
                            for f_name in feature_names:
                                f_value = feature[f_name] #[sid]
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
                                    f_value = f_value #[0] # every feature values are [x]
                                    fout_model_file.write('%s:%s '%(str(fid),str(f_value)))
                            fout_model_file.write('\n')

                        elif level == 'stories': # story-level
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
                                    f_value = f_value #[0] # every feature values are [x]
                                    fout_model_file.write('%s:%s '%(str(fid),str(f_value)))
                            fout_model_file.write('\n')

                        else:
                            print ('Wrong level:',level)
                            import pdb; pdb.set_trace()
                            sys.exit(1)

        print ('Done with %s\n'%(main_style_type))
    print ('Done\n')
    return



def load_dataset(data_dir, level='sentences'):
    datatypes = ['train','valid','test']
    datas = []
    for dt in datatypes:
        data = []
        files = glob.glob(os.path.join(data_dir, level, dt)+'/*.json')
        for file in files:
            with open(file) as fin:
                data.append(json.load(fin))
        print('Number of files/loaded-data: {}/{} in {}'.format(len(files),len(data), dt))
        datas.append(data)
    return tuple(datas)


def get_dic(data, verbose=False, limit=False):
    print ('Extracting dictionaries from data...')
    # persona dictionaries
    combined_dic = defaultdict(lambda: defaultdict(list))
    controlled_dic = defaultdict(list)

    for pid,obj in enumerate(data):
        if limit and pid > limit:
            break
        if verbose:
            print ('PID: %d'%(pid))

        sents = obj['output.sentences']
        # import pdb; pdb.set_trace()
        sents = [sents]
        styles = obj['persona']

        # (1) combined setting dictionary
        for s,v in styles.items():
            combined_dic[s][v] += sents

        # (2) controlled setting dictionary
        style_ordered_tuple = tuple([styles[s] for s in STYLE_ORDER])
        assert len(style_ordered_tuple) == len(STYLE_ORDER)
        controlled_dic[style_ordered_tuple] += sents
    print ('Combined Dic',combined_dic.keys())
    print ('Combined Combinations',' '.join(
        ['%s:%d'%(k,len(v)) for k,v in combined_dic.items()]))
    for style, value_dic in combined_dic.items():
        print ('- STYLE: %s, Value Types: %d' % (style,len(value_dic)))
        print (' * Value Dist: ',' '.join(['%s:%d'%(k,len(v)) for k,v in value_dic.items()]))

    print ('Controlled Dic',len(controlled_dic))
    return combined_dic, controlled_dic



def main(args, ngram_size=3, model_dir = '../../exp/model/', is_lower=True):
    project = args[1]
    data_dir = args[2]
    model_dir = args[3]
    w2v_dir = args[4]
    max_features = int(args[5]) # maximum number of ngram features (not vocab size)
    level = args[6]
    exp_setting = args[7]

    # (1) load dataset (json files)
    train,valid,test = load_dataset(data_dir,level=level)
    # combine train+valid+test for classification test
    data = train+valid+test

    # (2) extract per-persona sentences
    combined_dic, controlled_dic = get_dic(data) #, limit=500)

    # (3) feature engineering
    vect, ngram_vocab, vocab = get_lexical_features(
            data,level,
            ngram_size=ngram_size,
            max_features=max_features,
            is_lower=is_lower )
    emb = import_embeddings(w2v_dir, vocab, project)

    # (4) add features
    add_features(ngram_vocab, emb)

    # (5) Extracting features
    if exp_setting == 'combined':
        extract_features_combined_persona(
                combined_dic, vect=vect, ngram_vocab = ngram_vocab, emb=emb,
                model_dir=model_dir, level=level, is_lower=is_lower) #, max_sent=max_sent)
    elif exp_setting == 'controlled':
        extract_features_controlled_persona(
                controlled_dic, vect=vect, ngram_vocab = ngram_vocab, emb=emb,
                model_dir=model_dir, level=level, is_lower=is_lower) #, max_sent=max_sent)


if __name__ == '__main__': main(sys.argv)

