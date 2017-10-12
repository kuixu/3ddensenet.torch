#!/bin/bash
dir=/data/kuixu/data/cryoem/inSilico_PDB_5-1A
sort -R $dir/file.list |awk -v dir=$dir 'NR<=30{print >> dir"/data_train.txt"}NR>30{print >> dir"/data_val.txt"}'
