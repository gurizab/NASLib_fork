#!/bin/bash

train_sizes=(5 8 14 24 42 71 121 205 347 589 1000) #(1000) #(10 15 23 36 56 87 135 209 323 500)
searchspaces=(transbench101_micro transbench101_macro)
datasets=(jigsaw class_object class_scene autoencoder normal room_layout segmentsemantic)
start_seed=9000
n_seeds=10

experiment=$1

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for searchspace in "${searchspaces[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for size in "${train_sizes[@]}"
        do
            sbatch ./scripts/cluster/xgb_correlation/run.sh $searchspace $dataset $size $start_seed $experiment --bosch
        done

        echo ""
    done
done