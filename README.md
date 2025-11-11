## DeepSTARR-Mouse

DeepSTARR-Mouse is a Convolutional Neural Network (CNN) adapted from the previously published DeepSTARR architecture (Nature Genetics, 2022). This model is designed for use in a transfer-learning framework to predict enhancer activity in E11.5 mouse embryos. For each tissue, CNNs are pre-trained on DNA accessibility data (i.e., ATAC-seq) and fine-tuned on a limited set of experimentally validated enhancers (VISTA enhancer browser, [https://enhancer.lbl.gov/vista/](https://enhancer.lbl.gov/vista/)).

*Targeted Design of Mammalian Tissue-Specific Enhancers In Vivo*
Shenzhi Chen, Vincent Loubiere, Ethan W. Hollingsworth, Sandra H. Jacinto, Atrin Dizehchi, Jacob Schreiber, Evgeny Z. Kvon, Alexander Stark. 2025

This repository contains the code used to train the models, make predictions, and design tissue-specific enhancers by [Ledidi](https://www.biorxiv.org/content/10.1101/2025.04.22.650035v1).

## Sequence-to-accessibility Model training
Data were used for Sequence-to-accessibility model training are uploaded at HuggingFace (https://huggingface.co/datasets/Shenzhi-Chen/DeepSTARR-Mouse-dataset/accessibility_model_dataset).
To train and evaluate models across 3 cross-validation folds for 3 tissues (heart, limb, and midbrain/CNS), download the `accessibility_model_dataset` and run the following script:
```
Accessibility_model_training/Run_models.sh
```

This script trains 2 replicates for each of the 3 cross-validation folds across the 3 tissues, creating a total of 18 models. For each model, it generates predictions and computes nucleotide contribution scores on the held-out test dataset.

Outputs are saved in separate directories. For example, the output for the heart model from fold 1, replicate 1 is located at:
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
To train and evaluate models across 3 cross-validation folds for 3 tissues (heart, limb, and midbrain/CNS), download the `enhancer_activity_model_dataset` and run the following script:
```
Enhancer_activity_model_training/Run_models.sh
```

This script trains 2 replicates for each of the 3 cross-validation folds across the 3 tissues, creating a total of 18 models. For each model, it generates predictions and computes nucleotide contribution scores on the held-out test dataset.

Outputs are saved in separate directories. For example, the output for the heart model from fold 1, replicate 1 is located at:
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
To benchmark the performance of transfer learning, we implemented two alternative control strategies:
(i).  **Direct Training:** Models trained directly on annotated VISTA enhancers without a pre-training step.
(ii).  **Accessibility Model Predictions:** Predictions from the sequence-to-accessibility models, which are scaled to a [0, 1] range and used as a proxy for enhancer activity.
To run these control experiments, execute the following script:
```
Model_evaluation/run_control_models.sh
```
(i) The output for a model trained directly on enhancer data (e.g., for heart, fold 1, replicate 1) is saved at:
```
enhancer_activity_model/heart/results_fold01_heart_DeepSTARR_rep1_init_random

# Trained model
- Model.json # Model archeticture
- Model.h5 # Trained model weights

# Predictions on held-out test dataset
- fold01_sequences_test.fa_predictions_Model.txt
```

(ii) The enhancer activity predictions derived from the corresponding sequence-to-accessibility model (e.g., for heart, fold 1, replicate 1) are saved at:
```
accessibility_model/heart/results_fold01_heart_DeepSTARR_rep1/enhancer

# Enhancer predicted activity from the sequence‑to‑accessibility models
- fold01_sequences_test.fa_predictions_Model.txt
```

### Ledidi Tissue-Specific Enhancer Design

This section describes how to design novel tissue-specific enhancers using **Ledidi**, a model-guided gradient optimization approach ([Jacob et al., 2022](https://www.biorxiv.org/content/10.1101/2025.04.22.650035v1)).

#### Methodology

The design process starts with 1,200 random DNA sequences for each tissue. These sequences are generated with dinucleotide frequencies matched to known VISTA enhancers to ensure a realistic starting point.

*   **Reference Sequences:** The VISTA sequences used for frequency matching are available on Hugging Face: [testing_dataset](https://huggingface.co/datasets/Shenzhi-Chen/DeepSTARR-Mouse-dataset/tree/main/testing_dataset).

These seed sequences are then optimized using both the sequence-to-accessibility and sequence-to-activity models with the following parameters:
*   **Target Accessibility Score:** 12
*   **Target Activity Score:** 14
*   **Edit Penalty:** `0.1` (to limit the number and magnitude of edits)

For activity optimization, pre-sigmoid logits are used instead of the final sigmoid output to avoid gradient saturation and improve optimization stability.

#### Execution

To run the enhancer design process, execute the following script:
```
Ledidi_enhancer_design/run_ledidi_enhancer_design.sh
```
This script applies all 18 trained models (2 replicates x 3 folds x 3 tissues) to design enhancers. To manage memory, it processes the sequences in batches, generating 20 sequences per model in each of the 10 batches. This results in a total of **1,200 designed sequences** for each tissue.

The designed sequences are saved in separate directories. For example, the output for the heart model from fold 1, replicate 1, batch 1 is located at:
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

### Designing New Tissue-Specific Enhancers

To design your own tissue-specific enhancers for heart, limb, or CNS in the mouse embryo, follow the steps below.
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
* -i Path to the FASTA file used as a reference for random sequence generation (e.g., all_vista_seq.fa).
* -w The number of enhancers to design.
* -v Path to the sequence-to-activity model used for sequence enhancer activity optimization
* -x Path to the sequence-to-accessibility model used for sequence DNA accessibility optimization
* -a The directory where designed sequences will be saved.
* -o 1/0 Set to 0 to adjust dinucleotide frequencies of random sequences to match the reference, or 1 to skip adjustment.
* -y The target score for DNA accessibility optimization.
* -z The target score for enhancer activity optimization.
* -l A parameter to control the magnitude of edits. Smaller values result in fewer edits to the starting sequences.

## Questions
If you have any questions/requests/comments please contact me at [shenzhichen1999@gmail.com](mailto:shenzhichen1999@gmail.com).
