#!/bin/bash

######
# Shenzhi Chen (2024)
######

################################################################################
# Set default values
################################################################################

Script_dir=Accessibility_models/

################################################################################
# Help
################################################################################

if [ $# -eq 0 ]; then
  echo >&2 "
$(basename $0) - Train DeepSTARR single task model + contribution scores + TF-Modisco

USAGE: $(basename $0) -d <fold file name> -a <architecture> -v <cell type output> -o <results output path> -p <peaks to generate nucl contr scores> -c <1/0 run nucl contr scores and TF-Modisco> -t <tissue for training> -f <fold number>

 -d     Input ID to get fasta file and sequences txt file         [ required ]
 -a     Model architecture                                        [ required ]
 -v     Variable to predict                                       [ required ]
 -o     Output directory name                                     [ required ]
 -p     Peaks to compute nucl contr scores                       [ required ]
 -c     Run nucl contr scores and TF-Modisco(0/1)                 [ required ]
 -t     tissue for training                                      [ required ]
 -f     fold number                                             [ required ]
 
"
  exit 1
fi

################################################################################
# Parse input and check for errors
################################################################################

while getopts "d:a:v:o:p:c:t:f:" o
do
    case "$o" in
        d) Input_ID="$OPTARG";;
        a) arch="$OPTARG";;
        v) variable_output="$OPTARG";;
        o) OUTDIR="$OPTARG";;
        p) ContrScores_peaks="$OPTARG";;
        c) ContrScores="$OPTARG";;
        t) tissue="$OPTARG";;
        f) fold="$OPTARG";;
        \?) exit 1;;
  esac
done


echo
echo Input_fasta: ${Input_ID}
echo Architecture: ${arch}
echo variable_output: ${variable_output}
echo Output director: ${OUTDIR}
echo ContrScores_peaks: ${ContrScores_peaks}
echo ContrScores TF-Modisco: ${ContrScores}
echo tissue: ${tissue}
echo fold: ${fold}
echo

### create output directory
mkdir -p ${OUTDIR}

################################################################################
# Train model
################################################################################

echo
echo "Training model ..."
echo

mkdir -p ${OUTDIR}/log_training
bin/my_bsub_gridengine -P g -G "gpu:3" -m 200 -T '6:00:00' -o ${OUTDIR}/log_training -n Training_${Input_ID} "${Script_dir}/Train_model_weighted.py -i ${Input_ID} -w ${Weight} -v score -a ${arch} -o ${OUTDIR}/Model" > ${OUTDIR}/log_training/msg.model_training.tmp

# get job IDs to wait for mapping to finish
ID_main=$(paste ${OUTDIR}/log_training/msg.model_training.tmp | grep Submitted | awk '{print $4}')

################################################################################
# Predictions for evaluation
################################################################################

echo
echo "Predictions of sets for evaluation ..."
echo
# predict whole chr18 accessibility
# predict test set

pred_script=${Script_dir}/Predict_CNN_model_from_fasta.py
model=${OUTDIR}/Model
Input_ID_name=$(basename ${Input_ID})

scripts/functions/my_bsub_gridengine -n predict_testset_${Input_ID} \
                                     -d "${ID_main}" \
                                     -o ${OUTDIR}/log_predictions \
                                     -m 40 \
                                     -T '12:00:00' \
                                     "${pred_script} \
                                     -s ${Input_ID}_sequences_test.fa \
                                     -m ${model} \
                                     -o ${OUTDIR}"

################################################################################
# Nucleotide contribution scores
################################################################################

if [ "$ContrScores" == "1" ]; then
  echo
  echo "Calculating Nucleotide contribution scores ..."
  echo

  mkdir -p ${OUTDIR}/log_explainer

  # calculate contribution scores
  for seq in ${ContrScores_peaks//,/ }; do
    seq_name=$(basename ${seq})
    scripts/functions/my_bsub_gridengine -c "g1|g2|g3" \
                                         -d "${ID_main}" \
                                         -m 60 \
                                         -T '10:00:00' \
                                         -P g -G "gpu:3" \
                                         -o ${OUTDIR}/log_explainer \
                                         -n DeepExplainer_${seq_name} \
                                         "${Script_dir}/run_DeepSHAP_DeepExplainer_for_modiscolite.py \
                                         -i ${seq} \
                                         -p ${OUTDIR}/ \
                                         -o ${OUTDIR}/ \
                                         -m Model \
                                         -s ${seq_name} \
                                         -w 1001 \
                                         -b dinuc_shuffle" > ${OUTDIR}/log_explainer/msg.contr_scores_${seq_name}.tmp
  done

fi