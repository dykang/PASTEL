# PASTEL
Data and code for ["(Male, Bachelor) and (Female, Ph.D) have different connotations: Parallelly Annotated Stylistic Language Dataset with Multiple Personas"](https://arxiv.org/) by Dongyeop Kang, Varun Gangal, and Eduard Hovy, EMNLP 2019

## The PASTEL dataset
PASTEL is a parallelly annotated stylistic language dataset.
The dataset consists of ~41K parallel sentences and 8.3K parallel stories annotated across different personas.

#### Examples in PASTEL:
<img src="dataset.png" width="70%">

#### Style-transfer using PASTEL
<img src="transfer.png" width="60%">


## Setup Configuration
Run `./setup.sh` at the root of this repository to install dependencies, unzip the data file into data/ directory, and download GloVe embedding under data/word2vec/.

## Data Format, Examples, and Simple Loading Script
Once you extract the `data.zip` under `./data/` directory, you will see two sub-directories: `sentences` for single sentence level and `stories` for story level (i.e., a series of five sentences). Each sentence/story is split into the train/dev/test splits (same splits used in the paper). For each split, annotation files are stored as json files. Each json file for `stories`/`sentences` level is fomatted as follows:
 - **id**: ID
 - **annotation_id** & **per_annotation_id**: annotation ID and ID per annotation
 - **persona**: seven types of annotator's persona (country, politics, tod, age, education, ehtnic, gender)
 - **input.keywords**: a list (set) of keywords for story (sentence) (e.g., ['(pretty|grown|gress)', '(single|orange|flower)',..]
 - **input.images**: a list of (single) image links originally from the [ViST](http://visionandlanguage.net/VIST/) dataset for story (sentence)
 - **input.sentences**: a list of (single) reference sentence originally from the [ViST](http://visionandlanguage.net/VIST/) dataset for story (sentence)
 - **output.sentences**: a list of (single) annotated sentence for story (sentence)
 

Here are example json files for `stories` (top) and `sentences` (botton):
```
{'annotation_id': '22724', 
 'per_annotation_id': '22724', 
 'id': '22724_2_3', 
 'persona': {'country': 'U.S.A', 'politics': 'RightWing', 'tod': 'Night', 'age': '45-54', 'education': 'NoDegree', 'ethnic': 'Caucasian', 'gender': 'Female'}, 
 'input.keywords': '(partied|night)', 
 'input.images': 'https://farm1.staticflickr.com/43/82526956_6192dcfa33_o.jpg', 
 'input.sentences': 'and partied the night away', 
 'output.sentences': 'They partied all night.'}
```
```
{'annotation_id': '38918', 
 'per_annotation_id': '38918', 
 'id': '38918_1', 
 'persona': {'country': 'U.S.A', 'politics': 'LeftWing', 'tod': 'Afternoon', 'age': '18-24', 'education': 'Bachelor', 'ethnic': 'Caucasian', 'gender': 'Female'}, 
 'input.keywords': ['(pretty|grown|grass)', '(single|orange|flower)', '(world|tallest|buildings)', '(leaf|growing|ground)', '(perfect|flower|bed)'], 
 'input.images': ['https://farm1.staticflickr.com/180/446895070_c43b800121_o.jpg', 'https://farm1.staticflickr.com/204/446902039_805dbe086f_o.jpg', 'https://farm1.staticflickr.com/187/446504229_392ffe3b05_o.jpg', 'https://farm1.staticflickr.com/207/446505566_2bd71c2fcb_o.jpg', 'https://farm1.staticflickr.com/233/446900415_dcf59c4007_o.jpg'], 
 'input.sentences': ['over grown grass but the flowers are really pretty .', 'a single beautiful orange flower .', 'some of the tallest buildings in the world .', '4 leaf clover but they are growing off the ground .', 'the perfect flower bed .'], 
 'output.sentences': ['The grass had grown and was very pretty.', 'There was one single orange flower in the grass.', "Nearby were the world's tallest buildings.", 'There was a leaf growing in the ground.', 'The flower bed looked just perfect']}
```

To directly use our dataset for your applications, please use our example script:

```shell
  python code/examples/load_dataset.py
```




## Models
Our codes are written by Python 3.6. In order to experiment with (and hopefully improve) our models for two applications, please run following commands:

To run controlled style classification (i.e., given a text, predict a gender of it), you can run:

```shell
  cd ./code/StyleClassify/
  ./run_classify.sh
```

To run parallel style transfer (i.e., text1 + style -> text2), you can run:

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
