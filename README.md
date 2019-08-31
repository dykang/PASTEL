# PASTEL
Data and code for ["(Male, Bachelor) and (Female, Ph.D) have different connotations: Parallelly Annotated Stylistic Language Dataset with Multiple Personas"](https://arxiv.org/) by Dongyeop Kang, Varun Gangal, and Eduard Hovy, EMNLP 2019

## The PASTEL dataset
PASTEL is a parallelly annotated stylistic language dataset.
The dataset consists of ~41K parallel sentences and 8.3K parallel stories annotated across different personas.

Each story is split into the train/dev/test splits (same splits used in the paper). Due to licensing constraints, we provide instructions and scripts for downloading the VIST dataset for raw reference sentences and images.

#### Examples in PASTEL: 
<img src="dataset.png" width="90%" height="90%">

#### Style-transfer using PASTEL 
<img src="transfer.png" width="80%" height="80%">


## Setup Configuration
Run `./setup.sh` at the root of this repository to install dependencies and download the VIST dataset.

## Models
In order to experiment with (and hopefully improve) our models for controlled style classification (i.e., given a text, predict a gender of it) and parallel style transfer (i.e., text1 + style -> text2), you can run following commands:

```
    python XXX.py
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
