#!/bin/bash
set -euo pipefail

meta="replacement_transfer_learning/TL_labelling_training_meta.txt"

while IFS=',' read -r ID dataset Rule fold rep tissue output input_prefix_label input_prefix_fasta access_model vista_testset_fa vista_testset_label size; do
  # skip header
  if [[ "${ID}" == "ID" ]]; then
    continue
  fi

  mkdir -p "${output}/log"

  bin/my_bsub_gridengine \
    -c "g1|g2|g3" \
    -m 40 \
    -T '1:00:00' \
    -P g -G "gpu:1" \
    -o "${output}/log" \
    -e "${output}/log" \
    -n "training_${ID}_${tissue}_${fold}_${rep}" \
    "replacement_transfer_learning/Train_replacement_transfer_learning_model.py \
        -i ${fold} \
        -v class \
        -a ${access_model} \
        -o ${output}/Model \
        -p ${input_prefix_fasta} \
        -q ${input_prefix_label} \
        -t ${vista_testset_fa} \
        -u ${vista_testset_label} \
        -s ${size}"

done < "${meta}"