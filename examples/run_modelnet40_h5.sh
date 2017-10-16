#!/bin/bash
cd ../
t=`date +%Y%m%d%H%M%S`
devs=$1
C=`echo $devs|awk -F',' '{print NF}'`
##n=$(($C/2))
n=$C
echo "nGPU:"$n
bz=$[24 * $n ]

net=3ddensenet_modelnet
de=30
gr=16
modeldir="${net}_bz${bz}_de${de}_gr${gr}"

dataset="modelnet40_60x_h5"
datadir="/data/kuixu/data/"$dataset
nclass=40
#nclass=21
echo "batchSize:"$bz
epoch=300
logfilename=${t}_${modeldir}_${dataset}_n${n}.log
echo "log is saved into logs/$logfilename"
CUDA_VISIBLE_DEVICES=$devs th main.lua \
    -resetClassifier true \
    -retrain $datadir/${modeldir}/model_best.t7 \
    -debug 0 \
    -netType $net \
    -batchSize $bz \
    -LR 0.01 \
    -lossfunc cross \
    -nGPU $n \
    -nThreads $n \
    -optnet false \
    -dataset modelnet \
    -datatype 3dclf \
    -nClasses $nclass \
    -nEpochs $epoch \
    -manualSeed 23 \
    -resume $datadir/$modeldir \
    -save $datadir/$modeldir \
    -train_data $datadir/train.list \
    -test_data $datadir/test.list \
    -hdf true \
    -labelstart0 true \
    -growthRate $gr \
    -depth $de \
    #| tee logs/$logfilename
    #-shareGradInput true \
    #-gendata gen/emdb.t7 \
    #-ignorelabel 3 \ # last label
    #-gendata data/torch7/emdb-${da}.t7 \
echo "log is saved into logs/$logfilename"
