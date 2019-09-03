#!/bin/bash

#echo 'Installing python dependencies..'
#pip install -r requirements.txt
#python -m spacy download en
#python -m spacy download en_core_web_sm


echo 'Installing NLG-EVAL package..'
cd code/
git clone https://github.com/Maluuba/nlg-eval.git
cd nlg-eval/
python setup.py install
cd ../../

echo 'Zipping data files under data/'
# Unzipping data file
cd data/
unzip ./data.zip
cd ../

# download glove file
echo 'Downloading word2vec (i.e. Glove) embeddings..'
W2V_DIR=./data/word2vec/
mkdir -p ${W2V_DIR}
wget -N http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip -P ${W2V_DIR}
unzip ${W2V_DIR}/glove.840B.300d.zip -d ${W2V_DIR}/
python -m gensim.scripts.glove2word2vec -i ${W2V_DIR}/glove.840B.300d.txt -o ${W2V_DIR}/glove.840B.300d.w2v.txt

