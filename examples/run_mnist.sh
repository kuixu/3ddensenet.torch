#!/bin/bash
cd ../
t=`date +%Y%m%d%H%M%S`
devs=$1
C=`echo $devs|awk -F',' '{print NF}'`
##n=$(($C/2))
n=$C
echo "nGPU:"$n

## Data
dataset="mnist-h5"
datadir="data/"$dataset

bz=$[1000 * $n ]

net=cnn
nclass=10
echo "batchSize:"$bz
epoch=100
logfilename=${t}_${net}_${dataset}_n${n}_bz${bz}_e${epoch}.log
echo "log is saved into logs/$logfilename"
CUDA_VISIBLE_DEVICES=$devs time th main.lua \
    -debug 0 \
    -netType $net \
    -depth 20 \
    -batchSize $bz \
    -LR 0.1 \
    -lossfunc cross \
    -nGPU $n \
    -nThreads $n \
    -optnet false \
    -dataset image2d \
    -nClasses $nclass \
    -nEpochs $epoch \
    -manualSeed 23 \
    -save ${datadir}/${net} \
    -resume ${datadir}/${net} \
    -train_data ${datadir}/train.list \
    -test_data ${datadir}/test.list \
    -hdf true \
    | tee logs/$logfilename
    #-shareGradInput true \
    #-gendata gen/emdb.t7 \
    #-ignorelabel 3 \ # last label
    #-gendata data/torch7/emdb-${da}.t7 \
echo "log is saved into logs/$logfilename"
