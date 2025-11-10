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


MM_using_DeepMELs:
	This notebook shows how to load and use the provided models. 
	It shows how to calculate and plot:
	 	Predictions
	 	Deepexplainer contribution scores
	 	In silico saturation mutagenesis
	3 models are provided: DeepMEL, DeepMEL2, and DeepMEL2 with GABPA extension.
	These models can be downloaded from Zenodo, which are used by Kipoi database:
		DeepMEL: https://zenodo.org/records/3592129
		DeepMEL2: https://zenodo.org/records/4590308
		DeepMEL_GABPA: https://zenodo.org/records/4590405
	
	
MM_EFS:
	This notebook shows how to design synthetic sequences by using in silico evolution.
	It uses the selected enhancers from the MM_Cbust_Homer notebook
	It consists of:
		Generating GC-adjusted random sequences:
		Performing in silico evolution and random drift experiments.
		Plotting the findings.
		Printing generated DNA sequences in nucleotide letters.
	Luciferase values are in ./data/luciferase folder
	Intermediate files are saved to ./data/deepmel2 folder
	Figures are saved to ./figures/evolution_from_scratch


MM_EFS_TFModisco:
	This notebook shows the TFModiscco experiments.
	It uses the synthetic sequences file generated via MM_using_DeepMELs notebook. 
	It consists of:
		Calculating contribution scores on synthetic sequences.
		Performing TFModisco on contribution scores.
		Plotting identified patterns.
		Saving trimmed patterns as txt file to be later used for motif analysis.
	Result files are saved to ./data/tfmodisco folder
	Figures are saved to ./figures/tfmodisco folder
	
		
MM_EFS_Steps_Repressors:
	This notebook shows how to perform mutations on generated sequences and visualize mutational steps.
	It uses the synthetic sequences file generated via MM_using_DeepMELs notebook.
	It consists of:
		Printing DNA sequences in nucleotide letters for different mutational steps.
		Applying mutations to selected position and substation.
		Plotting the findings.
	Luciferase values are in ./data/luciferase folder
	Result files are saved to ./data/tfmodisco folder
	Figures are saved to ./figures/mutational_steps and ./figures/repressor_addition folders
	
	
MM_Enhance_Rescue:
	This notebook shows the near-enhancer and enhancing active enhancer experiments.
	Luciferase values are in ./data/enhance_rescue/luciferase folder
	Figures are saved to ./figures/enhance_rescue folder

	
MM_IRF4_Experiments:
	This notebook shows the experiments performed on IRF4 enhancer.
	It consists of:
		Loading the IRF4 enhancer sequence with different motif modifications.
		Loading saturation mutagenesis assay performed on IRF4 enhancer by Kircher et al.
		Showing individual mutations generating repressor binding sites on IRF4 enhancer.
		Plotting the findings. 
	In vitro saturation mutagenesis assay value file is in ./data/irf4/
	Luciferase values are in ./data/luciferase folder
	Figures are saved to ./figures/irf4 folder
	
	
MM_ZEB2_ChIP:
	This notebook shows the experiments related to ZEB2 ChIP-seq on MM001 cell line.
	Processed ZEB2 ChIP-seq (Antibody and input), ATAC-seq, and SOX10 ChIP-seq on MM001 files are in ./data/chip_seq
	ZEB2 ChIP-seq summit file is in ./data/chip_seq
	The notebook consists of:
		Plotting ZEB2 vs SOX10 ChIP-seq values compared with accessibility.
		Finding and plotting regions with high ZEB2 signal.
		Plotting ZEB2 and SOX10 ChIP-seq values on irf4 locus
	Figures are saved to ./figures/chip_seq folder


MM_Lenti_ATAC:
	This notebook shows the experiments related to ATAC-seq on synthetic enhancer integrated cell lines.
	Processed ATAC-seq data is in data/lenti_atac_chip folder.
	It consist of:
		Reading ATAC-seq files and calculating the coverage on the enhancers
	Figures are saved to ./figures/lenti_atac_chip folder


MM_ChromBPnet_Experiments:
	This notebooks shows scoring synthetic and genomic enhancer by using the ChromBPNet models trained on MM001 and MM047 cell lines.
	It uses the synthetic sequences file generated via MM_using_DeepMELs notebook. 
	The model files are provided in ./data/chrombpnet.
	Figures are saved to ./figures/chrombpnet folder.
	
	
MM_Enformer_Experiments:
	This notebooks shows scoring synthetic and genomic enhancer by using the Enformer model.
	Enformer model is loaded from "https://tfhub.dev/deepmind/enformer/1"
	Enformer class annotation is in ./data/enformer folder.
	It uses the synthetic sequences file generated via MM_using_DeepMELs notebook. 		
	The intermediate prediction files are in ./data/enformer folder.
	Figures are saved to ./figures/enformer folder.
	
	
MM_Motif_Implanting:
	This notebook shows how to design synthetic sequences by using motif implantation.
	It consists of:
		Performing motif implantation experiments.
		Visualising motif distance preference experiments.
		Replacing motifs on synthetic sequences with weaker ones from IRF4 enhancer.
		Cutting and shortening designed sequences.
	Luciferase values are in ./data/motif_embedding folder
	Intermediate files are saved to ./data/motif_embedding folder
	Figures are saved to ./figures/motif_embedding
	

MM_GAN:
	This notebook shows how to load and analyse GAN generated sequences.
	GAN generated sequences are provided in ./data/gan/generated_seqs folder.
	Background sequences are provided in ./data/gan/background_seqs folder.
	Genomic sequences are provided in ./data/gan folder
	It consists of:
		Reading GAN generated, genomic, and background sequences.
		Scoring generated sequences with the DeepMEL model.
		Visualising prediction scores on gan generated sequences at different training steps.
		Comparing GC content of GAN generated and background sequences.
		Visializing the luciferase results and contribution score plots.
	Luciferase values are in ./data/luciferase folder
	Intermediate files are saved to ./data/gan folder
	Figures are saved to ./figures/gan folder
	
	
MM_Cbust_Homer:
	This notebook shows ClusterBuster and Homer experiments.
	It uses contribution scores and TFModisco scores generated in the MM_EFS_TFModisco notebook.
	The motif database file is provided in ./data/tomtom folder
	It consists of:
		Getting TFModisco patterns and saving as txt file to be later used by ClusterBuster.
		Running Tomtom on TFModisco patterns.
		Running ClusterBuster by using TFModisco pattern PWMs on the sequences generated by in silico evolution, motif implantation, and GAN.
		Running Homer using Random and Evolved sequences as foreground and background sequences, and vice versa.
	ClusterBuster results are in ./data/cbust folder.
	Homer results are in ./data/homer folder.
	Figures are saved to ./figures/cbust folder
		