#!/bin/bash

declare -A styNo=(
    [STYLED]=2
    [ethnic]=2
    [gender]=2
    [age]=2
    [education]=2
    [country]=2
    [tod]=2
    [politics]=2
)

testName=test


modelName=simpleModelGlove2

#for sty in STYLED ethnic gender Country edu TOD
#STYLED ethnic gender education country tod politics
for sty in age
do
    echo "Running Inference For $sty"
    CUDA_VISIBLE_DEVICES=0 python main.py -mode=inference -method=beam -emb_size=300 -hidden_size=384 -modelName=tmp/${modelName}_${sty}_${styNo[$sty]}.ckpt -problem=$sty -no_mem_optimize | tee logs/Inference_${modelName}_${sty}_${styNo[$sty]}
    echo "Computing Metrics For $sty"
    python computeNLGEvalMetrics.py tmp/${modelName}_${sty}_${styNo[$sty]}.ckpt.test.beam.output data/${testName}.${sty}.tgt | tee logs/NLGEval_${modelName}_${sty}_${styNo[$sty]}
    echo "Done for $sty"
done

#exit

#modelName=simpleModelGlove2BothPretrained

#for sty in STYLED #Politics ethnic gender Country edu TOD
#do
    #echo "Running Inference For $sty"
    #CUDA_VISIBLE_DEVICES=1 python main.py -mode=inference -method=beam -emb_size=300 -hidden_size=384 -modelName=tmp/${modelName}_${sty}_${styNo[$sty]}.ckpt -problem=$sty | tee logs/Inference_${modelName}_${sty}_${styNo[$sty]}
    #echo "Computing Metrics For $sty"
    #python computeNLGEvalMetrics.py tmp/${modelName}_${sty}_${styNo[$sty]}.ckpt.test.beam.output data/${testName}.${sty}.tgt | tee logs/NLGEval_${modelName}_${sty}_${styNo[$sty]}
    #echo "Done for $sty"
#done
