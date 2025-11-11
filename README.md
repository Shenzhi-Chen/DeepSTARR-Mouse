## DeepSTARR-Mouse
DeepSTARR‑Mouse is a Convolutional Neural Network (CNN) adapted from the previously published DeepSTARR architecture (Nature Genetics, 2022). This model is designed for use in a transfer‑learning framework to predict enhancer activity in E11.5 mouse embryos. For each tissue, CNNs are pre‑trained on DNA accessibility data (i.e., ATAC‑seq) and fine‑tuned on a limited set of experimentally validated enhancers (VISTA enhancer browser, https://enhancer.lbl.gov/vista/).

*<ins>Targeted Design of Mammalian Tissue-Specific Enhancers In Vivo</ins>*  
Shenzhi Chen, Vincent Loubiere, Ethan W. Hollingsworth, Sandra H. Jacinto, Atrin Dizehchi, Jacob Schreiber, Evgeny Z. Kvon, Alexander Stark. 2025

This repository contains the code used to to train the models, make predictions and design tissue-specific enhancers by Ledidi (https://www.biorxiv.org/content/10.1101/2025.04.22.650035v1).

## Sequence-to-accessibility Model training
Data were used for Sequence-to-accessibility model training are uploaded at HuggingFace (https://huggingface.co/datasets/Shenzhi-Chen/DeepSTARR-Mouse-dataset/accessibility_model_dataset).
To train models across 3 Cross-validation folds for 3 tissues (heart, limb and midbrain(CNS))and evaluate them, download the training data (accessibility_model_dataset), run following script:
```
Accessibility_model_training/Run_models.sh
```

This script will train 2 replicates for each 3 Cross-validation folds of 3 tissues and you will get 18 models in sum, for each of them this scripts will make predictions and compute nucleotide contribution scores on held-out test dataset.

Outputs are speprately saved, as cross-validation fold 1 and replicate 1 for heart as a example, all outputs is saved under accessibility_model/heart/results_fold01_heart_DeepSTARR_rep1.
```
accessibility_model/heart/results_fold01_heart_DeepSTARR_rep1

# Trained model
- Model.json # Model archeticture
- Model.h5 # Trained model weights

# Predictions on held-out test dataset
- fold01_sequences_test.fa_predictions_Model.txt

# Nuceotide contribution score with sequences one-hot code
- fold01_sequences_test_onehot.npz # Sequence one-hot code 
- fold01_sequences_test_contrib.npz # Nucleotide contribution score
- Model_fold01_sequences_test.fa_dinuc_shuffle_deepSHAP_DeepExplainer_importance_scores.h5 # Combined h5 file
```

## Sequence-to-activity Model training
Data were used for Sequence-to-activity model training are uploaded at HuggingFace (https://huggingface.co/datasets/Shenzhi-Chen/DeepSTARR-Mouse-dataset/tree/main/enhancer_activity_model_dataset). Data were used for evaluation model are uploaded at HuggingFace (https://huggingface.co/datasets/Shenzhi-Chen/DeepSTARR-Mouse-dataset/tree/main/testing_dataset).
To train models across 3 Cross-validation folds for 3 tissues (heart, limb and midbrain(CNS))and evaluate them, download the training data (enhancer_activity_model_dataset), run following script:
```
Enhancer_activity_model_training/Run_models.sh
```

This script will train 2 replicates for each 3 Cross-validation folds of 3 tissues and you will get 18 models in sum, for each of them this scripts will make predictions and compute nucleotide contribution scores on held-out test dataset.

Outputs are speprately saved, as cross-validation fold 1 and replicate 1 for heart as a example, all outputs is saved under enhancer_activity_model/heart/results_fold01_heart_DeepSTARR_rep1.
```
enhancer_activity_model/heart/results_fold01_heart_DeepSTARR_rep1

# Trained model
- Model.json # Model archeticture
- Model.h5 # Trained model weights

# Predictions on held-out test dataset
- fold01_sequences_test.fa_predictions_Model.txt

# Nuceotide contribution score with sequences one-hot code
- fold01_sequences_test_onehot.npz # Sequence one-hot code 
- fold01_sequences_test_contrib.npz # Nucleotide contribution score
- Model_fold01_sequences_test.fa_dinuc_shuffle_deepSHAP_DeepExplainer_importance_scores.h5 # Combined h5 file
```

## Control model training
To compare with transfer learing, we applied two alternatives:
(i) models trained directly on annotated VISTA enhancers without pre‑training.
(ii) predictions from the sequence‑to‑accessibility models (scaled to [0, 1] and used as activity predictions).		
run following script:
```
Model_evaluation/run_control_models.sh
```
(i) As cross-validation fold 1 and replicate 1 directly trained model for heart as an example, is saved under enhancer_activity_model/heart/results_fold01_heart_DeepSTARR_rep1_init_random:
```
enhancer_activity_model/heart/results_fold01_heart_DeepSTARR_rep1_init_random

# Trained model
- Model.json # Model archeticture
- Model.h5 # Trained model weights

# Predictions on held-out test dataset
- fold01_sequences_test.fa_predictions_Model.txt
```

(ii) As cross-validation fold 1 and replicate 1 trained model for heart as an example, prediction from the sequence‑to‑accessibility model is saved under accessibility_model/heart/results_fold01_heart_DeepSTARR_rep1:
```
accessibility_model/heart/results_fold01_heart_DeepSTARR_rep1/enhancer

# Enhancer predicted activity from the sequence‑to‑accessibility models
- fold01_sequences_test.fa_predictions_Model.txt
```

## Ledidi tissue specific enhancer design
For each tissue, 1,200 random DNA sequences were generated with dinucleotide frequencies matched to VISTA sequences and used as input for Ledidi, a model‑guided gradient optimization approach (https://www.biorxiv.org/content/10.1101/2025.04.22.650035v1). To limit the number and magnitude of edits, the edit‑penalty parameter was set to 0.1. Seed sequences were then optimized using both the sequence‑to‑accessibility and sequence‑to‑activity models, with target predicted values set to 12 and 14, respectively. For activity optimization, the final sigmoid layer was removed and pre‑sigmoid logits were used to avoid gradient saturation and improve optimization stability. The reference sequences, which used to adjust dinucleotide frequencies were uploaded at HuggingFace (https://huggingface.co/datasets/Shenzhi-Chen/DeepSTARR-Mouse-dataset/tree/main/testing_dataset). Run following script:
```
Ledidi_enhancer_design/run_ledidi_enhancer_design.sh
```
This script apply trained 2 replicates for each 3 Cross-validation folds of 3 tissues model to design tissue specific enhancer. This script will randomly generate DNA sequences, and optimize both DNA accessibility and enhancer activity for targeted tissue, in each batch it will generate 20 sequences from single model to save memory usage and there are 10 batch, in sum this scripts will design 1,200 sequences for each tissue.

Outputs are speprately saved, as cross-validation fold 1 and replicate 1 for heart model batch 1 design as a example, designed sequences is saved under ledidi_design/heart/results_fold01_heart_DeepSTARR_rep1/1
```
ledidi_design/heart/results_fold01_heart_DeepSTARR_rep1/1

# 20 designed tissue specific enhancers
- 1adjusted12_14_designed_sequence_evolution.fasta 
```

## Prediction for new DNA sequences
To predict the accessibility levels or enhancer activity score in a given tissue of the mouse embryo for new DNA sequences, please run:
```
# Clone this repository
git https://github.com/Shenzhi-Chen/DeepSTARR-Mouse.git
cd DeepSTARR-Mouse

# download a Sequence-to-accessibility or Sequence-to-activity model from HuggingFace https://huggingface.co/Shenzhi-Chen/DeepSTARR-Mouse
# example with Sequence-to-accessibility model for heart fold01 replicate1 as accessibility_models/heart/results_fold01_heart_DeepSTARR_rep1/Model*

# create 'DeepSTARR' conda environment by running the following:
conda create --name DeepSTARR_Mouse python=3.7 tensorflow=2.4.1 keras=2.4.3 # or tensorflow-gpu/keras-gpu if you are using a GPU
source activate DeepSTARR_Mouse
pip install git+https://github.com/AvantiShri/shap.git@master
pip install 'h5py<3.0.0'
pip install deeplift==0.6.13.0

# Run prediction script on fasta files with 1,001 bp sequences
python Accessibility_model_training/Predict_CNN_model_from_fasta.py \
  -s Sequences_example.fa \
  -m accessibility_models/heart/results_fold01_heart_DeepSTARR_rep1/Model \
  -o Sequences_example

```

Where:
* -s FASTA file with input 1,001 bp DNA sequences
* -m model file (from accessibility or enhancer activity model)
* -o output directory

We recommend using the models from the different folds and average the prediction scores for a more robust prediction.

## Design new tissue-specific enhancers
To design tissue specific enhancers for your favourite tissue among three tissues (heart, limb and CNS) in mouse embryo, please run:
```
# Clone this repository
git https://github.com/Shenzhi-Chen/DeepSTARR-Mouse.git
cd DeepSTARR-Mouse

# download both Sequence-to-accessibility and Sequence-to-activity model for your favourite tissue from HuggingFace https://huggingface.co/Shenzhi-Chen/DeepSTARR-Mouse
# download reference VISTA sequences for randomly generated sequences dinucleotide frequency adjustment from HuggingFace https://huggingface.co/datasets/Shenzhi-Chen/DeepSTARR-Mouse-dataset/tree/main/testing_dataset
# example with heart fold01 replicate1 using accessibility_models/heart/results_fold01_heart_DeepSTARR_rep1/Model and enhancer_activity_models/heart/results_fold01_heart_DeepSTARR_rep1/Model 

# create 'DeepSTARR' conda environment by running the following:
conda create --name DeepSTARR_Mouse python=3.7 tensorflow=2.4.1 keras=2.4.3 # or tensorflow-gpu/keras-gpu if you are using a GPU
source activate DeepSTARR_Mouse
pip install git+https://github.com/AvantiShri/shap.git@master
pip install 'h5py<3.0.0'
pip install deeplift==0.6.13.0
pip install ledidi==2.1.0

# Run prediction script on fasta files with 1,001 bp sequences
python Ledidi_enhancer_design/ledidi_enhancer_design.py \
  -i testing_dataset/all_vista_seq.fa \
  -w 20 \
  -v enhancer_activity_models/heart/results_fold01_heart_DeepSTARR_rep1/Model \
  -x accessibility_models/heart/results_fold01_heart_DeepSTARR_rep1/Model \
  -a Sequences_design_example \
  -o 0 \
  -y 12 \
  -z 14 \
  -l 0.1
```

Where:
* -i FASTA file with 1,001 bp DNA sequences as reference for random sequence generation
* -w Number of designed sequences
* -v sequence-to-activity model used for enhancer activity optimization
* -x sequence-to-accessibility model used for DNA accessibility optimization
* -a output directory
* -o 1/0 Not adjust dinuceotide ratio/Adjust dinuceotide ratio for random generated sequences
* -y Targeted DNA accessibility score
* -z Targeted enhancer activity score
* -l Edit‑penalty parameter, The smaller this value the less edit on starting sequences

## Questions
If you have any questions/requests/comments please contact me at [shenzhichen1999@gmail.com](mailto:shenzhichen1999@gmail.com).
