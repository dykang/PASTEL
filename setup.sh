pip install -r requirements.txt
python -m spacy download en
python -m spacy download en_core_web_sm

cd code/
git clone https://github.com/Maluuba/nlg-eval.git
cd nlg-eval/
python setup.py install

# TODO download glove.840B.300d.txt in ./data/word2vec/

