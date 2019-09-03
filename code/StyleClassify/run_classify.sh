#!/bin/bash

PROJ='PASTEL'
DATADIR=../../data/
MODELDIR=../../model/
W2VDIR=../..//data/word2vec/glove.840B.300d.txt
MAXFEATURE=70000

LTYPES=('controlled' 'combined')
LEVELS=('sentences' 'stories')

for LEVEL in "${LEVELS[@]}"
    for LTYPE in "${LTYPES[@]}"
    do
        echo "=============================================="
        echo "Extracting features..." $PROJ $DATADIR $MODELDIR $MAXFEATURE $LEVEL $LTYPE
        echo "=============================================="
        python feature_extract.py \
            $PROJ $DATADIR $MODELDIR $W2VDIR \
            $MAXFEATURE $LEVEL $LTYPE

        ABLATION=True #False #True
        FCHOOSES=(False) #'deep' 'lexical' 'syntax' False)
        STYLES=('gender' 'age' 'education' 'politics')
        for FCHOOSE in "${FCHOOSES[@]}"
        do
            for STYLE in "${STYLES[@]}"
            do
                echo "=============================================="
                echo "Classifying..." $PROJ $DATADIR $MODELDIR $MAXFEATURE $STYLE $LTYPE $LEVEL $ABLATION $FCHOOSE $FCHOOSE
                echo "=============================================="
                python classify.py \
                    $PROJ $DATADIR $MODELDIR $MAXFEATURE \
                    $STYLE $LEVEL $LTYPE $ABLATION $FCHOOSE
            done
        done
    done
done

