
# directly training activity model (work as conrtol)
OUTDIR=result/model/enhancer_activity_model/${tissue}/results_${fold}_${tissue}_DeepSTARR_rep${rep}_init_random
mkdir -p ${OUTDIR}
mkdir -p ${OUTDIR}/log_training
scripts/functions/my_bsub_gridengine  -P g \
											                -G "gpu:1" \
											                -m 60 \
											                -T '2:00:00' \
											                -o ${OUTDIR}/log_training \
											                -n act_Training_init_random \
											                 "${script_path}/Train_random_initi_model.py  \
											                 -i ${InputID}/${tissue}/${fold} \
											                 -v class \
											                 -o ${OUTDIR}/Model" > ${OUTDIR}/log_training/msg.model_training.tmp
# get job IDs to wait for mapping to finish
ID_main=$(paste ${OUTDIR}/log_training/msg.model_training.tmp | grep Submitted | awk '{print $4}')
# Predict enhancer test dataset
mkdir -p ${OUTDIR}/log_prediction
JOB_ID=act_VISTA_pred_${ID}_init_random
scripts/functions/my_bsub_gridengine -n ${JOB_ID} \
                                     -o ${OUTDIR}/log_predictions \
                                     -m 80 \
                                     -d "${ID_main}" \
                                     -T '4:00:00' \
                                     "${script_path}/Predict_CNN_model_from_fasta.py \
                                     -s ${InputID}/${tissue}/${fold}_sequences_test.fa \
                                     -m ${OUTDIR}/Model \
                                     -o ${OUTDIR}"   
