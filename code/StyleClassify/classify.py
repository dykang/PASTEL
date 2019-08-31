# -*- coding: utf-8 -*-
import os
import sys
import pickle
import operator
import random
import glob
import ntpath
from collections import Counter
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn import datasets,preprocessing,model_selection
from sklearn import linear_model,svm,neural_network,ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from .utils import import_embeddings

def read_feature_names(filename):
    feature_dic = {}
    feature_list = []
    with open(filename) as fin:
        for line in fin:
            tks = line.strip().split('\t')
            feature_dic[int(tks[0].strip())] = tks[1].strip()
            feature_list.append(tks[1].strip())
    return feature_dic,feature_list

def read_class_names(filename):
    class_list = []
    with open(filename) as fin:
        for line in fin:
            tks = line.strip().split('\t')
            class_list.append(tks[0].strip())
    return class_list



def get_data(features_if, scale=False, n_features = None, split=True, shuffle=True, ratio=0.2, feature_ablation=False):
    x,y = datasets.load_svmlight_file(features_if, n_features=n_features) #,multilabel=True)
    if feature_ablation:
        if len(feature_ablation) < 1000:
            x = x[:, feature_ablation]
        else:
            x_mask = np.array([True]*x.shape[1])
            x_mask[feature_ablation] = False
            x = x.todense()[:,x_mask]
    if scale:
        x = preprocessing.scale(x.toarray())
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=ratio, random_state=42)
    return x_train, y_train, x_test, y_test



def classify(model_dir,feature_filename, feature_list_filename, class_filename, ablation, feature_ablation = False):

    ablation_out_filename = 'ablation/ablation_%s.txt' %(ntpath.basename(feature_filename).replace('svmlite.',''))
    log_out_filename = 'log/log_%s.txt' %(ntpath.basename(feature_filename).replace('svmlite.',''))

    fout = open(log_out_filename, 'a')
    fout.write(feature_filename+'\n')

    ###########################
    # data loading
    ###########################

    feature_dic, feature_list= read_feature_names(model_dir + feature_list_filename)
    print ('Num of features:',len(feature_list))
    classes = read_class_names(model_dir + class_filename)
    print ('Num of classes:',len(classes))

    if feature_ablation:
        print('Ablation range:',feature_ablation)

    n_features = None # sum(1 for line in open(args[5])) #None #train_features.shape[1]
    train_features, train_labels, test_features, test_labels = \
            get_data(
                model_dir+feature_filename,
                scale=False, n_features=n_features,
                split=True, shuffle=True, ratio=0.2, feature_ablation=feature_ablation)

    ###########################
    # majority
    ###########################
    train_counter = Counter(train_labels)
    test_counter = Counter(test_labels)
    print (train_counter, train_features.shape)
    print (test_counter, test_features.shape)

    majority_clf =DummyClassifier(strategy="most_frequent")
    majority_clf.fit(train_features, train_labels, )

    train_y_hat = majority_clf.predict(train_features)
    train_p, train_r, train_f, _ = precision_recall_fscore_support(train_labels, train_y_hat, average='macro')
    print ('Majority P/R/F: %.2f / %.2f / %.2f in %d examples' %(
            train_p*100.0, train_r*100.0, train_f*100.0, sum(train_labels)))
    fout.write('Majority(Tr)\t%.2f\t%.2f\t%.2f\t%d\n'%(
            train_p*100.0, train_r*100.0, train_f*100.0, sum(train_labels)))

    # test
    test_y_hat = majority_clf.predict(test_features)
    test_p, test_r, test_f, _ = precision_recall_fscore_support(test_labels, test_y_hat, average='macro')
    print ('Majority P/R/F: %.2f / %.2f / %.2f in %d examples' %(
            test_p*100.0, test_r*100.0, test_f*100.0, sum(test_labels)))
    fout.write('Majority(Ts)\t%.2f\t%.2f\t%.2f\t%d\n'%(
            test_p*100.0, test_r*100.0, test_f*100.0, sum(test_labels)))

    train_major_p = train_p
    train_major_r = train_r
    test_major_p = test_p
    test_major_r = test_r

    ###########################
    #classifiers
    ###########################
    clfs = []
    if not ablation:
        for c in [.1, .25, .5, 1.0]:
            for clf in [
                    linear_model.LogisticRegression(C=c, dual=True),
                    linear_model.LogisticRegression(C=c, penalty='l1'),
                    svm.SVC(kernel='rbf', C=c)
                ]:
                clfs.append(clf)
        clfs += [
            neural_network.MLPClassifier(alpha=1),
            ensemble.AdaBoostClassifier()]
    else:
        for c in [.1, .25, .5, 1.0]:
            for clf in [
                    linear_model.LogisticRegression(C=c, dual=True),
                ]:
                clfs.append(clf)
    random.shuffle(clfs)
    print ('Total number of classifiers',len(clfs))

    ###########################
    # training (CV) and testing
    ###########################
    best_classifier = None
    best_v = 0
    for cidx, clf in enumerate(clfs):
        scores = model_selection.cross_val_score(clf, train_features, train_labels, cv=5, n_jobs=6)
        v = sum(scores)*1.0/len(scores)
        if v > best_v:
            #print("New best v!",v*100.0,clf)
            best_classifier = clf
            best_v = v
    print("Best v:",best_v*100.0,", Best clf: ",best_classifier)
    best_classifier.fit(train_features, train_labels)

    fout.write('Best %s\n'%(best_classifier))


    ###########################
    # ablation
    ###########################
    if ablation is True:
        fout_ablation = open(ablation_out_filename, 'w')
        fout_ablation.write(feature_filename+'\n')

        for cid in range(len(classes)):
            cls = classes[cid]
            # import pdb; pdb.set_trace()
            assert len(classes) == 2
            coef = best_classifier.coef_[0]

            assert len(coef) == len(feature_list)
            cf_list = zip(coef, feature_list)


            if cid ==0 :
                cf_sorted = sorted(cf_list, key = lambda x:-x[0])
            else:
                cf_sorted = sorted(cf_list, key = lambda x:x[0])
            #cf_sorted = [(c,f) for c,f in cf_sorted] #    if not f.startswith('emb')
            cf_sorted = [(c,f) for c,f in cf_sorted]

            print ('Class: ',cls, ', '.join(['[%s x %s]'%(f,round(c,2)) for c,f in cf_sorted[:20]]))

            fout_ablation.write('%s\n'%(cls))
            cnt = 0
            max_features = 100
            for c,f in cf_sorted:
                if f.startswith('emb'): continue
                fout_ablation.write('\t%s\t%.2f\n'%(f,c))
                cnt += 1
                if cnt > max_features: break
        fout_ablation.write('\n')
        fout_ablation.close()


    # train
    train_y_hat = best_classifier.predict(train_features)
    train_p, train_r, train_f, _ = precision_recall_fscore_support(train_labels, train_y_hat, average='macro')
    print ('Train P/R/F: %.2f / %.2f / %.2f in %d examples' %(
            train_p*100.0, train_r*100.0, train_f*100.0, sum(train_labels)))
    fout.write('Train\t%.2f\t%.2f\t%.2f\t%d\n'%(
            train_p*100.0, train_r*100.0, train_f*100.0, sum(train_labels)))

    # test
    test_y_hat = best_classifier.predict(test_features)
    test_p, test_r, test_f, _ = precision_recall_fscore_support(test_labels, test_y_hat, average='macro')
    print ('Test P/R/F: %.2f / %.2f / %.2f in %d examples' %(
            test_p*100.0, test_r*100.0, test_f*100.0, sum(test_labels)))
    fout.write('Test\t%.2f\t%.2f\t%.2f\t%d\n'%(
            test_p*100.0, test_r*100.0, test_f*100.0, sum(test_labels)))
    fout.close()
    print ('Done\n')

    return (train_major_p,train_major_r), (test_major_p,test_major_r), (train_p,train_r), (test_p, test_r)



def cross_feature_joint(a_files, fn,model_dir):
    f_dic = defaultdict(lambda: defaultdict(list)) #defaultdict(list)
    for f in a_files:
        print ('reading..',f)
        c = None
        with open(f,'r') as fout:
            for idx, line in enumerate(fout):
                if idx==0: continue
                if line.startswith('\t'):
                    tks = line.strip().split('\t')
                    if not tks[0].startswith('emb'):
                        f_dic[c][tks[0]] += [float(tks[1])]
                else:
                    c = line.strip()

    fout = open(os.path.join(model_dir,fn),'w')

    classes = list(f_dic.keys())
    max_features = 20
    min_duplicate_combination = 3
    num_duplicate_words_bettwen_classes = 10

    f_tuples_sorted = {}
    for cid,c in enumerate(classes):
        # print (c)
        f_tuples = [(f,len(vs),np.average(vs)) for f,vs in f_dic[c].items()
                if len(vs) >= min_duplicate_combination]

        if    cid == 0: reverse=True
        else: reverse=False
        f_tuples.sort(key=operator.itemgetter(2), reverse=reverse )

        f_tuples_sorted[c] = f_tuples

    # filtering by looking each other's tuples not making overlapped
    for cid,c in enumerate(classes):
        f_tuples = f_tuples_sorted[c]
        f_tuples_other = f_tuples_sorted[classes[cid-1]]

        print (cid,c,len(f_tuples))
        cnt = 0
        fout.write(c+'\n')
        for f,l,a in f_tuples:
            if f in [f2 for f2,_,_ in f_tuples_other[:num_duplicate_words_bettwen_classes]]:
                continue
            cnt += 1
            if cnt > max_features : break
            print ('\t%s\t%d\t%.4f'%(f,l,a))
            fout.write('\t%s\t%d\t%.4f\n'%(f,l,a))



def main(args):
    project = args[1]
    data_dir = args[2]
    model_dir = args[3]
    max_feature = int(args[4]) # maximum number of ngram features (not vocab size)
    style = args[5]                         # 'gender', 'age' ....
    level = args[6]                # sentences or stories
    exp_setting = args[7]                # or 'single' or 'tuple'
    ablation = True if args[8].lower()=='true' else False
    feature_choose = False if args[9].lower()=='false' else args[9].lower()

    f_idxs = None
    if feature_choose:
        feature_choose_types = ['lexical', 'deep', 'syntax']
        #for f_type in feature_choose_types:

        if feature_choose    not in feature_choose_types:
            print ('Wrong feature choose type',feature_choose)
            sys.exit(1)

        f_type = feature_choose
        print ('-------------------------')
        print ('Choose only', f_type)
        print ('-------------------------')
        f_idxs = []
        if f_type == 'lexical':
            f_idxs = range(70000)
        elif f_type == 'deep':
            f_idxs = range(70000,70000+300)
        elif f_type == 'syntax':
            f_idxs = range(70000+300,70000+300+14)



    if exp_setting == 'combined':
        feature_filename = '%s/svmlite.%s_%s_%s_%d'%(exp_setting,style,exp_setting,level,max_feature)
        feature_list_filename = 'features.dat'
        class_filename = '%s/labels.%s_%s_%s_%d.txt'%(exp_setting,style,exp_setting,level,max_feature)

        classify(
                model_dir,
                feature_filename,
                feature_list_filename,
                class_filename,
                ablation=ablation,
                feature_ablation = f_idxs)

    elif exp_setting == 'controlled':
        feature_list_filename = 'features.dat'
        feature_filename_list = sorted(glob.glob(model_dir+ '%s/svmlite.%s_*_%s_%s_%d'%(exp_setting,style,exp_setting,level,max_feature)))
        class_filename_list = sorted(glob.glob(model_dir+ '%s/labels.%s_*_%s_%s_%d.txt'%(exp_setting,style,exp_setting,level,max_feature)))

        assert len(feature_filename_list) == len(class_filename_list) > 0
        print ('Total number of combination of external features',len(feature_filename_list))

        train_m_ps,train_m_rs = [], []
        test_m_ps,test_m_rs = [],[]
        train_ps, train_rs = [],[]
        test_ps, test_rs = [],[]

        for feature_filename, class_filename in zip(feature_filename_list,class_filename_list):
            (train_m_p,train_m_r), (test_m_p,test_m_r), (train_p, train_r), (test_p, test_r) =\
                    classify(model_dir, feature_filename,
                            feature_list_filename, class_filename, ablation=ablation,
                            feature_ablation = f_idxs)
            train_m_ps.append(train_m_p)
            train_m_rs.append(train_m_r)
            test_m_ps.append(test_m_p)
            test_m_rs.append(test_m_r)
            train_ps.append(train_p)
            train_rs.append(train_r)
            test_ps.append(test_p)
            test_rs.append(test_r)


        fout = open('log/log.%s_%s_%s_%d_%s.txt'%(style,exp_setting,level,max_feature,feature_choose), 'w')
        fout.write('%s_%s_%s_%d_%s\n'%(style,exp_setting,level,max_feature,feature_choose))

        print ('Macro-Averaged Majority Precision\t%.2f'%(100.0 * np.average(test_m_ps)))
        print ('Macro-Averaged Majority Recall\t%.2f'%(100.0 * np.average(test_m_rs)))
        print ('Macro-Averaged Majority F-Score\t%.2f'%(100.0 * 2 * np.average(test_m_ps) * np.average(test_m_rs) /
                (np.average(test_m_ps) + np.average(test_m_rs))))

        print ('Macro-Averaged Precision\t%.2f'%(100.0 * np.average(test_ps)))
        print ('Macro-Averaged Recall\t%.2f'%(100.0 * np.average(test_rs)))
        print ('Macro-Averaged F-Score\t%.2f'%(100.0 * 2 * np.average(test_ps) * np.average(test_rs) /
                (np.average(test_ps) + np.average(test_rs))))


        fout.write('Macro-Averaged Majority Precision\t%.2f\n'%(100.0 * np.average(test_m_ps)))
        fout.write('Macro-Averaged Majority Recall\t%.2f\n'%(100.0 * np.average(test_m_rs)))
        fout.write('Macro-Averaged Majority F-Score\t%.2f\n'%(100.0 * 2 * np.average(test_m_ps) * np.average(test_m_rs) /
                (np.average(test_m_ps) + np.average(test_m_rs))))

        fout.write('Macro-Averaged Precision\t%.2f\n'%(100.0 * np.average(test_ps)))
        fout.write('Macro-Averaged Recall\t%.2f\n'%(100.0 * np.average(test_rs)))
        fout.write('Macro-Averaged F-Scor\t%.2f\n'%(100.0 * 2 * np.average(test_ps) * np.average(test_rs) /
                (np.average(test_ps) + np.average(test_rs))))
        fout.close()


        if ablation:
            ablation_files = ['%s.txt'%(ntpath.basename(f).replace('svmlite.','ablation/ablation_')) for f in feature_filename_list]
            ablation_fn = 'ablation/ablation_%s_%s_%s_%d_%s.txt'%(style,exp_setting,level,max_feature,feature_choose)
            cross_feature_joint(ablation_files, ablation_fn, model_dir)
            print ('ablation done')



if __name__ == '__main__': main(sys.argv)

