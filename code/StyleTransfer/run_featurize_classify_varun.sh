#!/bin/bash
#set -x

PROJ='STYLeD'
MODELDIR='../../exp/model/'
STYLES=('gender' 'age' 'ethnic' 'Country' 'edu' 'Politics' 'TOD')

MAXFEATURE=70000
LEVELS=(1 ) #1) #1 # 0 for sentence, 1 for story
LTYPES=('single' 'multi') #'multi' #'single' # 'single' 'multi'



#for LEVEL in "${LEVELS[@]}"
#do
#  for LTYPE in "${LTYPES[@]}"
#  do
#    start_time=`date +%s`
#    echo "=============================================="
#    echo "Extracting features..." $PROJ $MODELDIR $MAXFEATURE $LEVEL $LTYPE
#    echo "=============================================="
#    python feature_extract.py $PROJ $MODELDIR $MAXFEATURE $LEVEL $LTYPE
#    echo "run-time: $(expr `date +%s` - $start_time) s"
#
#    ABLATION=False #True #False #True
#    start_time=`date +%s`
#    for STYLE in "${STYLES[@]}"
#    do
#      echo "=============================================="
#      echo "Classifying..." $PROJ $MODELDIR $MAXFEATURE $STYLE $LTYPE $LEVEL $ABLATION
#      echo "=============================================="
#      python classify.py \
#        $PROJ $MODELDIR $MAXFEATURE \
#        $STYLE $LTYPE $LEVEL $ABLATION
#    done
#    echo "run-time: $(expr `date +%s` - $start_time) s"
#  done
#done

python feature_extract_varun.py $PROJ $MODELDIR $MAXFEATURE 0 'single'
