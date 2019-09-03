# PASTEL
Data and code for ["(Male, Bachelor) and (Female, Ph.D) have different connotations: Parallelly Annotated Stylistic Language Dataset with Multiple Personas"](https://arxiv.org/) by Dongyeop Kang, Varun Gangal, and Eduard Hovy, EMNLP 2019

## The PASTEL dataset
PASTEL is a parallelly annotated stylistic language dataset.
The dataset consists of ~41K parallel sentences and 8.3K parallel stories annotated across different personas.
Each story is split into the train/dev/test splits (same splits used in the paper). The raw image links and sentences are extracted from the [ViST](http://visionandlanguage.net/VIST/) dataset.


#### Examples in PASTEL:
<img src="dataset.png" width="90%" height="90%">

#### Style-transfer using PASTEL
<img src="transfer.png" width="80%" height="80%">


## Setup Configuration
Run `./setup.sh` at the root of this repository to install dependencies, unzip the data file into data/ directory, and download GloVe embedding under data/word2vec/.

## A script to load the dataset quickly
Run ```python code/examples/load_dataset.py```

## Models
In order to experiment with (and hopefully improve) our models for two applications:


To run controlled style classification (i.e., given a text, predict a gender of it), you can run following commands:

```shell
  cd ./code/StyleClassify/
  ./run_classify.sh
```


To run parallel style transfer (i.e., text1 + style -> text2), you can run following commands:

```shell
  cd ./code/StyleClassify/
  ./run_transfer.sh
```




## Citation

    @inproceedings{kang19bemnlp,
      title = {(Male, Bachelor) and (Female, Ph.D) have different connotations: Parallelly Annotated Stylistic Language Dataset with Multiple Personas},
      author = {Dongyeop Kang and Varun Gangal and Eduard Hovy},
      booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      address = {Hong Kong},
      month = {November},
      url = {https://arxiv.org/},
      year = {2019}
    }

## Acknowledgement
 - This work would not have been possible without the ViST dataset and helpful suggestions with Ting-Hao Huang. We also thank Alan W Black, Dan Jurafsky, Wei Xu, Taehee Jung, and anonymous reviewers for their helpful comments.
