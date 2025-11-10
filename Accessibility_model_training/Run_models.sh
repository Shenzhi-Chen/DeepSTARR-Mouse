#!/bin/bash
tissue_list="midbrain,heart,limb"
fold_list="fold01,fold02,fold03"
script_path=accessibility_models
arch=DeepSTARR2
InputID=accessibility_model_dataset
OUTDIR=accessibility_model

TFModisco=1

for fold in ${fold_list//,/ }; do
    for tissue in ${tissue_list//,/ }; do
        for rep in 1 2; do
            mkdir ${OUTDIR}/${tissue}
  		  	${script_path}/Train_DeepSTARR_and_interpretation.sh \
    		    -d ${InputID}/${tissue}/${fold} \
    		  	-a ${arch} \
                -v score \
    		  	-o ${OUTDIR}/${tissue}/results_${fold}_${tissue}_${arch}_rep${rep} \
                -p ${InputID}/${tissue}/${fold}_sequences_test.fa \
    		  	-c $TFModisco \
                -t $tissue \
                -f $fold
        done
  	done
done


