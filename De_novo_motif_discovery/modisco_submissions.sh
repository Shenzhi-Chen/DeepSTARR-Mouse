#!/bin/bash

source activate tf_modisco_light 
module load meme/5.1.1-foss-2018b-python-3.6.6

FOLD=fold01
REP=rep1

#v10nr_clust_public
#JASPAR_VERT_FILE=/groups/stark/nikolaus.mandlburger/Projects/blastoid_project/res/accessibility_models_and_data/TE_total/dataselection_3/feature_attributions/trainingruns_280524/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt
JASPAR_VERT_FILE=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/src/modico_analysis/JASPAR2024_CORE_vertebrates_non-redundant_pfms_TN5_meme.txt
ADD_TFNAME_SCRIPT=/groups/stark/nikolaus.mandlburger/Scripts/DNN_modeling/add_tfname_to_modiscoreport.py


for TISSUE in heart limb midbrain;do

    SEQS=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/ATAC_models/${FOLD}_${REP}_sequences_onehot.npz
    CONTRIBS=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/ATAC_models/${FOLD}_${REP}_sequences_contrib.npz
    H5_OUT=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/ATAC_models/${FOLD}_${REP}_modisco.h5
    MODISCO_OUTDIR=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/ATAC_models/${FOLD}_${REP}_modisco/
    #$MYBSUB -m 60 -o /groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/ATAC_models/log -n ACC_${TISSUE}_modisco_acc -T 6:20:00 "modisco motifs -s ${SEQS} -a ${CONTRIBS} -n 50000 -o $H5_OUT && modisco report -i ${H5_OUT} -o ${MODISCO_OUTDIR} -m ${JASPAR_VERT_FILE} && $ADD_TFNAME_SCRIPT -r ${MODISCO_OUTDIR}motifs.html"
    $MYBSUB -m 60 -o /groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/ATAC_models/log -n ACC_${TISSUE}_modisco_acc -T 1:20:00 "modisco report -i ${H5_OUT} -o ${MODISCO_OUTDIR} -m ${JASPAR_VERT_FILE} && $ADD_TFNAME_SCRIPT -r ${MODISCO_OUTDIR}motifs.html"

    SEQS=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/VISTA_models/${FOLD}_${REP}_sequences_onehot.npz
    CONTRIBS=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/VISTA_models/${FOLD}_${REP}_sequences_contrib.npz
    H5_OUT=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/VISTA_models/${FOLD}_${REP}_modisco.h5
    MODISCO_OUTDIR=/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/VISTA_models/${FOLD}_${REP}_modisco/
    #$MYBSUB -m 60 -o /groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/VISTA_models/log -n ${TISSUE}_modisco_act -T 6:20:00 "modisco motifs -s ${SEQS} -a ${CONTRIBS} -n 50000 -o $H5_OUT && modisco report -i ${H5_OUT} -o ${MODISCO_OUTDIR} -m ${JASPAR_VERT_FILE} && $ADD_TFNAME_SCRIPT -r ${MODISCO_OUTDIR}motifs.html"
    $MYBSUB -m 60 -o /groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/${TISSUE}/VISTA_models/log -n ${TISSUE}_modisco_act -T 1:20:00 "modisco report -i ${H5_OUT} -o ${MODISCO_OUTDIR} -m ${JASPAR_VERT_FILE} && $ADD_TFNAME_SCRIPT -r ${MODISCO_OUTDIR}motifs.html"
done



