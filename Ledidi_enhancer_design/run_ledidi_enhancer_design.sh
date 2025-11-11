#!/bin/bash
referenece=testing_dataset/all_vista_seq.fa
tissues="limb,heart,midbrain"
fold="fold01,fold02,fold03"
rep="1,2"
TFModisco=1
script_path=Ledidi_enhancer_design

loop="1,2,3,4,5,6,7,8,9,10"
acc="12"
act="14"
lamda="0.1"

for fold in ${fold_list//,/ }; do
  for tissue in ${tissue_list//,/ }; do
    for rep in 1 2; do
      for i in ${loop//,/ }; do
      enhancer=enhancer_activity_models/${tissue}/results_${fold}_${tissue}_DeepSTARR_rep${rep}/Model
      access=accessibility_models/${tissue}/results_${fold}_${tissue}_DeepSTARR_rep${rep}/Model
      OUTDIR=ledidi_design/${tissue}/${i}

      mkdir ${OUTDIR}
      mkdir ${OUTDIR}/log_enhancer_design
      scripts/functions/my_bsub_gridengine -m 10 \
                                           -n design_adjust_${tissues}_${i} \
                                           -o ${OUTDIR}/log_enhancer_design \
                                           -T '2:00:00' \
                                           -P g -G "gpu:1" \
                                           "${script_path}/ledidi_enhancer_design.py \
                                           -i ${referenece} \
                                           -w 20 \
                                           -v ${enhancer} \
                                           -x ${access} \
                                           -a ${OUTDIR} \
                                           -o 0 \
                                           -y ${acc} \
                                           -z ${act} \
                                           -l ${lamda} "
    done
   done
  done
done
