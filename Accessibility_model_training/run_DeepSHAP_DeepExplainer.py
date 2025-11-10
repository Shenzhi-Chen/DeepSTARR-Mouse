"""
DeepSHAP Analysis for DNA Sequence Models

Author: Shenzhi Chen
"""

import os
import sys
import getopt
import h5py
import numpy as np

# TensorFlow and Keras imports
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
from keras.models import model_from_json

# DeepLIFT and SHAP imports
import shap
from deeplift.dinuc_shuffle import dinuc_shuffle

# Custom helper modules
sys.path.append('bin/')
from helper import IOHelper, SequenceHelper

# Set random seed for reproducibility
np.random.seed(1234)

# ============================================================================
# Global Configuration Parameters
# ============================================================================

# Number of background sequences to take an expectation over
NS = 50

# Number of dinucleotide shuffled sequences per sequence as background
DINUC_SHUFFLE_N = 50

# Default input DNA sequence length
INPUT_LENGTH = 1001


# ============================================================================
# Command Line Argument Parsing
# ============================================================================

def parse_arguments(argv):
    """
    Parse command line arguments for the DeepSHAP analysis script.
    
    Args:
        argv (list): Command line arguments
        
    Returns:
        tuple: Parsed arguments (data_path, model_path, out_path, model_ID, 
               sequence_set, input_length, bg)
               
    Raises:
        SystemExit: If required arguments are missing or invalid
    """
    model_ID = ''
    data_path = ''
    model_path = ''
    out_path = ''
    sequence_set = ''
    input_length = INPUT_LENGTH
    bg = ''
    
    try:
        opts, args = getopt.getopt(
            argv,
            "hi:p:o:m:s:w:b:",
            ["data=", "path=", "outpath=", "model=", "sequence_set=", 
             "input_length=", "bg="]
        )
    except getopt.GetoptError:
        print('Usage: run_DeepLIFT.py -i <path to data> -p <path to model> '
              '-o <path to result folder> -m <CNN model file> '
              '-s <sequence set> -w <input_length> -b <random/dinuc_shuffle>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: run_DeepLIFT.py -i <path to data> -p <path to model> '
                  '-o <path to result folder> -m <CNN model file> '
                  '-s <sequence set> -w <input_length> -b <random/dinuc_shuffle>')
            sys.exit()
        elif opt in ("-i", "--data"):
            data_path = arg
        elif opt in ("-p", "--path"):
            model_path = arg
        elif opt in ("-o", "--outpath"):
            out_path = arg
        elif opt in ("-m", "--model"):
            model_ID = arg
        elif opt in ("-s", "--sequence_set"):
            sequence_set = arg
        elif opt in ("-w", "--input_length"):
            input_length = int(arg)
        elif opt in ("-b", "--bg"):
            bg = arg
            
    # Validate required arguments
    if not model_ID:
        sys.exit("Error: CNN model file not specified")
    if not sequence_set:
        sys.exit("Error: sequence_set not specified")
    if not bg:
        sys.exit("Error: background method not specified (use 'random' or 'dinuc_shuffle')")
    if not data_path:
        sys.exit("Error: data path not specified")
    if not model_path:
        sys.exit("Error: model path not specified")
    if not out_path:
        sys.exit("Error: output path not specified")
        
    # Print configuration
    print('=' * 70)
    print('DeepSHAP Analysis Configuration')
    print('=' * 70)
    print(f'Input sequence path:     {data_path}')
    print(f'Input model path:        {model_path}')
    print(f'Output path:             {out_path}')
    print(f'CNN model file:          {model_ID}')
    print(f'Sequence set:            {sequence_set}')
    print(f'Input sequence length:   {input_length}')
    print(f'Background method:       {bg}')
    print('=' * 70)
    
    return data_path, model_path, out_path, model_ID, sequence_set, input_length, bg


# ============================================================================
# Sequence Processing Functions
# ============================================================================

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    """
    Fill a zeros array with one-hot encoded DNA sequence.
    
    Args:
        zeros_array (np.ndarray): Array to fill with one-hot encoding
        sequence (str): DNA sequence string
        one_hot_axis (int): Axis along which to encode (0 or 1)
        
    Note:
        This function mutates zeros_array in place.
        Encoding: A=0, C=1, G=2, T=3, N=skip
    """
    assert one_hot_axis in [0, 1], "one_hot_axis must be 0 or 1"
    
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)
    
    # Nucleotide to index mapping
    nuc_to_idx = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 
                  'G': 2, 'g': 2, 'T': 3, 't': 3}
    
    for i, char in enumerate(sequence):
        if char in ['N', 'n']:
            continue  # Leave position as all zeros
        elif char not in nuc_to_idx:
            raise RuntimeError(f"Unsupported character: {char}")
            
        char_idx = nuc_to_idx[char]
        
        if one_hot_axis == 0:
            zeros_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zeros_array[i, char_idx] = 1


def prepare_input(sequence_set_name, data_input_path, target_length):
    """
    Load and prepare DNA sequences for model input.
    
    Args:
        sequence_set_name (str): Name of the sequence set
        data_input_path (str): Path to FASTA file
        target_length (int): Target sequence length
        
    Returns:
        np.ndarray: One-hot encoded sequences (n_sequences, length, 4)
    """
    print(f'\nPreparing input for: {sequence_set_name}')
    
    # Load sequences from FASTA file
    input_fasta_data = IOHelper.get_fastas_from_file(
        str(data_input_path), 
        uppercase=True
    )
    
    target_length = int(target_length)
    
    # Pad sequences if necessary
    if len(input_fasta_data.sequence.iloc[0]) < target_length:
        print("Padding sequences at 3' end to match target length")
        print(f"Before padding: {len(input_fasta_data.sequence.iloc[0])} bp")
        
        padded_sequences = []
        for seq in input_fasta_data.sequence:
            diff = target_length - len(seq)
            if diff > 0:
                # Pad with random nucleotides
                padding = ''.join(np.random.choice(['C', 'G', 'T', 'A'], diff))
                padded_seq = seq + padding
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
            
        input_fasta_data.sequence = padded_sequences
        print(f"After padding: {len(input_fasta_data.sequence.iloc[0])} bp")
    
    # Get sequence length
    sequence_length = len(input_fasta_data.sequence.iloc[0])
    
    # Convert sequences to one-hot encoding
    seq_matrix = SequenceHelper.do_one_hot_encoding(
        input_fasta_data.sequence, 
        sequence_length, 
        SequenceHelper.parse_alpha_to_seq
    )
    
    # Replace NaN with zero and infinity with large finite numbers
    X = np.nan_to_num(seq_matrix)
    
    print(f"Prepared input shape: {X.shape}")
    return X


def load_model(model_path, model_id):
    """
    Load a Keras model from JSON architecture and H5 weights.
    
    Args:
        model_path (str): Path to directory containing model files
        model_id (str): Model identifier (filename without extension)
        
    Returns:
        tuple: (keras_model, keras_model_weights_path, keras_model_json_path)
    """
    keras_model_weights = os.path.join(model_path, f'{model_id}.h5')
    keras_model_json = os.path.join(model_path, f'{model_id}.json')
    
    # Load model architecture from JSON
    with open(keras_model_json, 'r') as json_file:
        keras_model = model_from_json(json_file.read())
    
    # Load model weights
    keras_model.load_weights(keras_model_weights)
    
    print('\nModel Summary:')
    print(keras_model.summary())
    
    return keras_model, keras_model_weights, keras_model_json


# ============================================================================
# DeepSHAP Explainer Functions
# ============================================================================

def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
    """
    Generate multiple dinucleotide-shuffled versions of a sequence.
    
    Dinucleotide shuffling preserves dinucleotide frequencies while
    randomizing the sequence, providing a biologically relevant background.
    
    Args:
        list_containing_input_modes_for_an_example (list): List containing 
            one-hot encoded sequence
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List containing array of shuffled sequences
    """
    assert len(list_containing_input_modes_for_an_example) == 1
    onehot_seq = list_containing_input_modes_for_an_example[0]
    
    rng = np.random.RandomState(seed)
    shuffled_seqs = np.array([
        dinuc_shuffle(onehot_seq, rng=rng) 
        for _ in range(DINUC_SHUFFLE_N)
    ])
    
    return [shuffled_seqs]  # Wrap in list for compatibility


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    """
    Combine multipliers and difference-from-reference to compute hypothetical
    contribution scores.
    
    This function computes what the importance scores would look like if
    different bases were present at each position, using the multipliers
    computed from the original sequence.
    
    Args:
        mult (list): Multipliers from DeepLIFT
        orig_inp (list): Original input sequence (one-hot encoded)
        bg_data (list): Background data
        
    Returns:
        list: Projected hypothetical contributions
    """
    assert len(orig_inp) == 1
    assert len(orig_inp[0].shape) == 2
    
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
    
    # Iterate over each nucleotide position
    for i in range(orig_inp[0].shape[-1]):
        # Create hypothetical input with only one nucleotide type
        hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")
        hypothetical_input[:, i] = 1.0
        
        # Compute difference from reference
        hypothetical_difference_from_reference = (
            hypothetical_input[None, :, :] - bg_data[0]
        )
        
        # Compute hypothetical contributions
        hypothetical_contribs = hypothetical_difference_from_reference * mult[0]
        
        # Sum across nucleotide axis and project
        projected_hypothetical_contribs[:, :, i] = np.sum(
            hypothetical_contribs, 
            axis=-1
        )
    
    return [np.mean(projected_hypothetical_contribs, axis=0)]


def compute_deepshap_scores(model, one_hot_sequences, background_method):
    """
    Compute DeepSHAP importance scores for DNA sequences.
    
    Args:
        model (keras.Model): Trained Keras model
        one_hot_sequences (np.ndarray): One-hot encoded sequences
        background_method (str): Background method ('random' or 'dinuc_shuffle')
        
    Returns:
        tuple: (hypothetical_scores, contribution_scores)
            - hypothetical_scores: Hypothetical contribution scores
            - contribution_scores: Actual contribution scores (projected onto input)
    """
    out_layer = -1  # Use last layer
    
    if background_method == "random":
        # Use random subset of sequences as background
        np.random.seed(seed=1111)
        background = one_hot_sequences[
            np.random.choice(one_hot_sequences.shape[0], NS, replace=False)
        ]
        explainer = shap.DeepExplainer(
            (model.layers[0].input, model.layers[out_layer].output),
            data=background,
            combine_mult_and_diffref=combine_mult_and_diffref
        )
    elif background_method == "dinuc_shuffle":
        # Use dinucleotide shuffling as background
        explainer = shap.DeepExplainer(
            (model.layers[0].input, model.layers[out_layer].output),
            data=dinuc_shuffle_several_times,
            combine_mult_and_diffref=combine_mult_and_diffref
        )
    else:
        raise ValueError(
            f"Invalid background method: {background_method}. "
            "Use 'random' or 'dinuc_shuffle'"
        )
    
    # Compute SHAP values
    shap_values_hypothetical = explainer.shap_values(one_hot_sequences)
    
    # Project hypothetical scores onto actual input bases
    # This gives the contribution of each base that is actually present
    shap_values_contribution = shap_values_hypothetical[0] * one_hot_sequences
    
    return shap_values_hypothetical[0], shap_values_contribution


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    data_path, model_path, out_path, model_ID, sequence_set, input_length, bg = \
        parse_arguments(sys.argv[1:])
    
    # Load sequences and model
    print("\n" + "=" * 70)
    print("Loading sequences and model...")
    print("=" * 70)
    
    X_all = prepare_input(sequence_set, data_path, input_length)
    keras_model, keras_model_weights, keras_model_json = load_model(model_path, model_ID)
    
    print(f"\nTotal sequences loaded: {X_all.shape[0]}")
    print(f"Sequence length: {X_all.shape[1]}")
    print(f"One-hot encoding dimension: {X_all.shape[2]}")
    
    # Compute DeepSHAP scores
    print("\n" + "=" * 70)
    print("Computing DeepSHAP importance scores...")
    print("=" * 70)
    
    scores = compute_deepshap_scores(keras_model, X_all, bg=bg)
    print(f"Computed {len(scores)} score types (hypothetical and contribution)")
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)
    
    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    
    # Save as HDF5
    h5_filename = os.path.join(
        out_path,
        f"{model_ID}_{sequence_set}_{bg}_deepSHAP_DeepExplainer_importance_scores.h5"
    )
    
    # Remove existing file if present
    if os.path.isfile(h5_filename):
        os.remove(h5_filename)
    
    with h5py.File(h5_filename, 'w') as f:
        # Save actual contribution scores
        g_contrib = f.create_group("contrib_scores")
        g_contrib.create_dataset("class", data=scores[1])
        print(f"Saved contribution scores")
        
        # Save hypothetical contribution scores
        g_hyp = f.create_group("hyp_contrib_scores")
        g_hyp.create_dataset("class", data=scores[0])
        print(f"Saved hypothetical contribution scores")
    
    print(f"Saved HDF5 file: {h5_filename}")
    
    # Reshape and save as NPZ format
    X_reshaped = X_all.swapaxes(1, 2)
    contrib_scores_reshaped = scores[1].swapaxes(1, 2)
    
    basename = os.path.basename(data_path).replace(".fa", "")
    
    # Save one-hot encoded sequences
    onehot_filename = os.path.join(out_path, f"{basename}_onehot.npz")
    np.savez(onehot_filename, X_reshaped)
    print(f"Saved one-hot sequences: {onehot_filename}")
    
    # Save contribution scores
    contrib_filename = os.path.join(out_path, f"{basename}_contrib.npz")
    np.savez(contrib_filename, contrib_scores_reshaped)
    print(f"Saved contribution scores: {contrib_filename}")
    
    print("\n" + "=" * 70)
    print("Analysis completed successfully!")
    print("=" * 70)