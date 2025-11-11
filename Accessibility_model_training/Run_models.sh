#!/bin/bash
tissue_list="midbrain,heart,limb"
fold_list="fold01,fold02,fold03"
script_path=accessibility_models
arch=DeepSTARR
InputID=accessibility_model_dataset
OUTDIR=accessibility_model
Contribution_score=1 #if compute nucleotide contribution score

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
    		  	-c $Contribution_score \
                -t $tissue \
                -f $fold
        done
  	done
done


