#!/bin/bash
tissue_list="midbrain,heart,limb"
fold_list="fold01,fold02,fold03"
script_path=Enhancer_activity_model_training
InputID=enhancer_activity_model_dataset
TFModisco=1
for fold in ${fold_list//,/ }; do
  for tissue in ${tissue_list//,/ }; do
    for rep in 1 2; do

          arch=result/model/accessibility_model/${tissue}/results_${fold}_${tissue}_DeepSTARR2_rep${rep}
					OUTDIR=result/model/enhancer_activity_model/${tissue}/results_${fold}_${tissue}_DeepSTARR2_rep${rep}

          mkdir ${OUTDIR}
					mkdir -p ${OUTDIR}/log_training
					JOB_ID=VISTA_${tissue}
					scripts/functions/my_bsub_gridengine -P g \
                                     -G "gpu:1" \
                                     -m 60 \
                                     -T '3:00:00' \
                                     -o ${OUTDIR}/log_training \
                                     -n transfer_learning_${JOB_ID} "${script_path}/Train_transfer_learning_model.py \
                                     -i ${InputID}/${tissue}/${fold} \
                                     -v class \
                                     -a ${arch}/Model \
                                     -o ${OUTDIR}/Model" > ${OUTDIR}/log_training/msg.model_training.tmp

					# Predict enhancer test dataset
					mkdir -p ${OUTDIR}/log_prediction
					JOB_ID=act_VISTA_pred_${ID}_init_random
					scripts/functions/my_bsub_gridengine -n ${JOB_ID} \
                                     -o ${OUTDIR}/log_predictions \
                                     -m 80 \
                                     -T '4:00:00' \
                                     "${script_path}/Predict_CNN_model_from_fasta.py \
                                     -s ${InputID}/${tissue}/${fold}_sequences_test.fa \
                                     -m ${OUTDIR}/Model \
                                     -o ${OUTDIR}"  
                                     
					# Predict random sequences
					scripts/functions/my_bsub_gridengine -n act_predict_random_${Input_ID}_init_random \
                                     -o ${OUTDIR}/log_predictions \
                                     -m 80 \
                                     -T '4:00:00' \
                                     "${script_path}/Predict_CNN_model_from_fasta.py \
                                     -s testing_dataset/random_sequences_600k.fasta \
                                     -m ${OUTDIR}/Model \
                                     -o ${OUTDIR}"                                      

          # contribution scores
         scripts/functions/my_bsub_gridengine -m 60 \
                                       -T '10:00:00' \
                                       -P g -G "gpu:3" \
                                       -o ${OUTDIR}/${tissue}/results_${fold}_${tissue}_${arch}_rep${rep}/log_explainer \
                                       -n ${tissue}/${fold}_DeepExplainer \
                                       "Accessibility_model_training/run_DeepSHAP_DeepExplainer.py \
                                       -i ${InputID}/${tissue}/${fold}_sequences_test.fa \
                                       -p ${OUTDIR}/ \
                                       -o ${OUTDIR}/ \
                                       -m Model \
                                       -s ${fold}_sequences_test.fa \
                                       -w 1001 \
                                       -b dinuc_shuffle" > ${OUTDIR}/log_explainer/msg.contr_scores_${fold}.tmp
  	    done
    done
done

