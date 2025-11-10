Dependencies:
    DL Python environment to use DeepMEL, DeepMEL2, and DeepFlyBrain:
    python=3.7 tensorflow-gpu=1.15 numpy=1.19.5 matplotlib=3.1.1 shap=0.29.3 ipykernel=5.1.2 h5py=2.10.0 TF-MoDISco 0.5.5.4

    DL Python environment to train GAN models:
    python=3.6 tensorflow-gpu=1.14.0 keras-gpu=2.2.4 numpy=1.16.2 matplotlib=3.1.1 shap=0.29.3 ipykernel=5.1.2


Deepexplainer script update:
    In order to calculate nucleotide contribution scores for only the selected class,
    conda_env/lib/python3.7/site-packages/shap/explainers/_deep/deep_tf.py is updated by inserting the following codes at line 277:
        elif output_rank_order.isnumeric():
            model_output_ranks = np.argsort(-model_output_values)
            model_output_ranks[0] = int(output_rank_order)


FLY_using_DeepFlyBrain:
	This notebook shows how to load and use the provided model. 
	It shows how to calculate and plot:
	 	Predictions
	 	Deexplainer contribution scores
	 	In silico saturation mutagenesis
	DeepFlyBrain is provided in ./models/deepflybrain
	The model can be downloaded from Zenodo, which is used by Kipoi database:
		DeepFlyBrain: https://zenodo.org/records/5153337

	
FLY_KC_EFS:
	This notebook shows how to design synthetic sequences by using in silico evolution for Kenyon Cells.
	It uses the selected enhancers from the MM_Cbust_Homer_Motif notebook
	It consists of:
		Generating GC-adjusted random sequences:
		Performing in silico evolution and random drift experiments.
		Plotting the findings.
		Printing generated DNA sequences in nucleotide letters.
	Intermediate files are saved to ./data/deepflybrain folder
	Figures are saved to ./figures/evolution_from_scratch
	

FLY_PNG_EFS:
	This notebook shows how to design synthetic sequences by using in silico evolution for Glial Cells.
	It uses the selected enhancers from the FLY_KC_EFS notebook
	It consists of:
		Performing in silico evolution experiments.
		Plotting the findings.
		Printing generated DNA sequences in nucleotide letters.
	Intermediate files are saved to ./data/deepflybrain folder
	Figures are saved to ./figures/evolution_from_scratch_PNG
	

FLC_KC_EFS_Steps_Rescue:
	This notebook shows how to visualize mutational steps and to get sequences with additional mutations.
	It uses the synthetic sequences file generated via FLY_KC_EFS notebook.
	It consists of:
		Printing DNA sequences in nucleotide letters for different mutational steps.
		Applying mutations to selected position and substation.
		Plotting the findings.
	Figures are saved to ./figures/mutational_steps and ./figures/rescue folders
	
	
FLY_KC_EFS_Mutation_Combination:
	This notebooks shows using alternative state space searches during in silico evolution
	It uses the synthetic sequences file generated via FLY_KC_EFS notebook.
	It consists of:
		Choosing top 20 best mutations instead of only top 1 during in silico evolution
		Investigating different evolution paths
		Choosing 5 random mutations instead following the model's guidance
	Intermediate files are saved to ./data/mutation_combination folder
	Figures are saved to ./figures/mutation_combination
	
	
FLY_KC_EFS_Mutation_Combination_All3Mut:
	This notebooks shows how to generate and score sequences with all possible 3 mutations
	It uses the synthetic sequences file generated via FLY_KC_EFS notebook.
	It consists of:
		Generating sequences with all possible 3 mutations
		Comparing prediction scores
	Figures are saved to ./figures/mutation_combination
	
	
FLY_KC_Near_Enhancer_Seqs:
	This notebook shows the in silico evolution of near-enhancer sequences.
	Kenyon Cell accessibility bigwig file is provided in ./data/near_enhancer_seq 
	Chopped fly genome is provided in ./data/near_enhancer_seq 
	It consists of:
		Calculating predictions on the 500bp chopped genomic sequences
		Plotting prediction scores vs chromatin accessibility
		Choosing sequences with low accessibility and high prediction score
		Plotting ATAC-seq coverage on chosen regions
		Performing additional in silico evolution mutations on the chosen regions
		Applying mutations to selected position and substation to create repressor binding sites
	Intermediate files are saved to ./data/near_enhancer_seq folder
	Figures are saved to ./figures/near_enhancer_seq folder
	
	
FLY_KC_Repressors:
	This notebook shows how to perform mutations on generated sequences and visualize mutational steps.
	It uses the synthetic sequences file generated via FLY_KC_EFS notebook.
	It consists of:
		Printing DNA sequences in nucleotide letters for different mutational steps.
		Applying mutations to selected position and substation.
		Plotting the findings.
	Figures are saved to ./figures/repressors
	

FLY_KC_ATAC:
	This notebook shows the experiments related to ATAC-seq on the brains of synthetic enhancer integrated fly lines.
	Processed ATAC-seq data is in ./data/atac folder.
	It consist of:
		Reading ATAC-seq files and calculating the coverage on the enhancers
	Figures are saved to ./figures/atac folder
	

FLY_EFS_TFModisco:
	This notebook shows the TFModiscco experiments.
	It uses the synthetic sequences file generated via FLY_KC_EFS notebook. 
	It consists of:
		Calculating contribution scores on synthetic sequences.
		Performing TFModisco on contribution scores.
		Plotting identified patterns.
		Saving trimmed patterns as txt file to be later used for motif analysis.
	Result files are saved to ./data/tfmodisco folder
	Figures are saved to ./figures/tfmodisco folder


FLY_Augmentation_Pruning:
	This notebook shows the experiments related to dual-code enhancers
	The cloned enhancers fasta file from Janssens et al is provided in ./data/augmentation_pruning
	The notebook consists of:
		Performing mutations on genomic enhancers to add a second code
		Identifying genomic enhancers accessible in two or more cell lines
		Performing mutations on genomic enhancers to remove the second code
	Figures are saved to ./figures/augmentation_pruning folder
		

FLC_KC_Motif_Implanting:
	This notebook shows how to design synthetic sequences by using motif implantation.
	It consists of:
		Performing motif implantation experiments.
		Visualising motif distance and location preference experiments.
		Identify enriched flankings at the motif implanted locations.
		Cutting designed sequences.
		Adding repressors sites by single mutations
		Replacing the background sequence of an enhancer with 1 million random sequences		
	Intermediate files are saved to ./data/motif_embedding folder
	Figures are saved to ./figures/motif_embedding
	
	
FLY_GAN:
	This notebook shows how to load and analyse GAN generated sequences.
	GAN generated sequences are provided in ./data/gan/generated_seqs folder.
	Background sequences are provided in ./data/gan/background_seqs folder.
	Genomic sequences are provided in ./data/gan folder
	It consists of:
		Reading GAN generated, genomic, and background sequences.
		Scoring generated sequences with the DeepMEL model.
		Visualising prediction scores on gan generated sequences at different training steps.
		Comparing GC content of GAN generated and background sequences.
		Visualising the results and contribution scores.
	Intermediate files are saved to ./data/gan folder
	Figures are saved to ./figures/gan folder
	
	
FLY_Cbust_Homer_Motif:
	This notebook shows ClusterBuster and Homer experiments.
	It uses contribution scores and TFModisco scores generated in the FLY_EFS_TFModisco notebook.
	The motif database file is provided in ./data/tomtom folder
	It consists of:
		Getting TFModisco patterns and saving as txt file to be later used by ClusterBuster.
		Running Tomtom on TFModisco patterns.
		Running ClusterBuster by using TFModisco pattern PWMs on the sequences generated by in silico evolution, motif implantation, and GAN.
		Running Homer using Random and Evolved sequences as foreground and background sequences, and vice versa.
	ClusterBuster results are in ./data/cbust folder.
	Homer results are in ./data/homer folder.
	Figures are saved to ./figures/cbust folder
