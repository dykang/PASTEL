#!/bin/bash
#set -x

PROJ='STYLeD'
DATADIR=../AMT/AMTResults/mainExp/trainNew/batches_StoryPlusKeywords2/
MODELDIR=../exp/model/
OUTDIR=../data
TRANSFERDIR=StyleTransfer/data/

python create_dataset_from_annotation.py \
  $PROJ $DATADIR $MODELDIR $OUTDIR $TRANSFERDIR



