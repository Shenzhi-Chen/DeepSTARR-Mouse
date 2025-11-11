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
Data were used for Sequence-to-activity model training are uploaded at HuggingFace (https://huggingface.co/datasets/Shenzhi-Chen/DeepSTARR-Mouse-dataset/tree/main/enhancer_activity_model_dataset).
To train models across 3 Cross-validation folds for 3 tissues (heart, limb and midbrain(CNS))and evaluate them, download the training data (enhancer_activity_model_dataset), run following script:
```
Enhancer_activity_model_training/Run_models.sh
```




