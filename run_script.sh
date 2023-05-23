#!/bin/bash

# /******************************************
# *MIT License
# *
# *Copyright (c) [2021] [Eleonora D'Arnese, Emanuele Del Sozzo, Davide Conficconi,  Marco Domenico Santambrogio]
# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"), to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# ******************************************/

           
IMG_DIM=512

PYCODE_Powell=powell_torch.py
PYCODE_oneplusone=one_plus_one_torch.py
PYCODE_oneplusone_mod=one_plus_one_modified.py
PYCODE_oneplusone_Heph=one_plus_one.py

DATASET_FLDR=./
CT_PATH=Dataset/ST0/SE0
PET_PATH=Dataset/ST0/SE4
GOLDEN_PATH=Dataset/ST0/NuovoGold
RES_PATH=Dataset/OutputClassicMoments

#metric=( MI CC MSE )
metric=(MI)
#dev=( cpu cuda )
dev=(cuda)

for i in "${metric[@]}"
do
    for j in "${dev[@]}"
    do
        #OLD
        #  echo "python $PYCODE_Powell -pt 1 -o 0 -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/powell_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -mtr $i -dvc $j"
        #  python3 $PYCODE_Powell -pt 1 -o 0 -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/powell_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -mtr $i -dvc $j
        #  echo "python $PYCODE_oneplusone -pt 1 -o 0 -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/oneplusone_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -mtr $i -dvc $j"
        #  python3 $PYCODE_oneplusone -pt 1 -o 0 -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/oneplusone_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -mtr $i -dvc $j 
       
        #new
        # echo "python3 $PYCODE_oneplusone_mod -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/powell_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -dvc $j -vol 246"
        # python3 $PYCODE_oneplusone_mod -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/oneplusone_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -dvc $j -vol 246
        # python3 res_extraction.py -f 0 -rg $GOLDEN_PATH/ -rt $RES_PATH/oneplusone_${i}_${j}/ -l oneplusone_${i}_${j} -rp ./
        # python3 AVGcompute.py -f gold-oneplusone_${i}_${j}-score_results.csv
        echo "python3 $PYCODE_Powell -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/powell_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -dvc $j -vol 246 -f Powell_classicMoments_mem_constr_dwarf2.csv"
        python3 $PYCODE_Powell -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/powell_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -dvc $j -vol 246
        python3 res_extraction.py -f 0 -rg $GOLDEN_PATH/ -rt $RES_PATH/powell_${i}_${j}/ -l powell_${i}_${j} -rp ./
        python3 AVGcompute.py -f gold-powell_${i}_${j}-score_results.csv
# # 
        echo "python $PYCODE_oneplusone -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/oneplusone_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -dvc $j -vol 246 -f 1p1_classicMoments_mem_constr_dwarf2.csv"
        python3 $PYCODE_oneplusone -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/oneplusone_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -dvc $j -vol 246
        python3 res_extraction.py -f 0 -rg $GOLDEN_PATH/ -rt $RES_PATH/oneplusone_${i}_${j}/ -l oneplusone_${i}_${j} -rp ./
        python3 AVGcompute.py -f gold-oneplusone_${i}_${j}-score_results.csv
# # 
# 
    done
done
