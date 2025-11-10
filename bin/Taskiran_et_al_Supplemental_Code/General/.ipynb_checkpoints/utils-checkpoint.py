import tensorflow as tf
import numpy as np
import matplotlib
from collections import Counter

def one_hot_encode_along_row_axis(sequence):
    to_return = np.zeros((1, len(sequence), 4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return[0], sequence=sequence, one_hot_axis=1)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)
    for (i, char) in enumerate(sequence):
        if char == "A" or char == "a":
            char_idx = 0
        elif char == "C" or char == "c":
            char_idx = 1
        elif char == "G" or char == "g":
            char_idx = 2
        elif char == "T" or char == "t":
            char_idx = 3
        elif char == "N" or char == "n":
            continue
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        if one_hot_axis == 0:
            zeros_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zeros_array[i, char_idx] = 1


def readfile(filename):
    ids = []
    ids_d = {}
    seqs = {}
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    seq = []
    for line in lines:
        if line[0] == '>':
            ids.append(line[1:].rstrip('\n'))
            id_line = line[1:].rstrip('\n').split('_')[0]
            if id_line not in seqs:
                seqs[id_line] = []
            if id_line not in ids_d:
                ids_d[id_line] = id_line
            if seq:
                seqs[ids[-2].split('_')[0]] = ("".join(seq))
            seq = []
        else:
            seq.append(line.rstrip('\n').upper())
    if seq:
        seqs[ids[-1].split('_')[0]] = ("".join(seq))

    return ids, ids_d, seqs


def prepare_data(filename):
    ids, ids_d, seqs, = readfile(filename)
    X = np.array([one_hot_encode_along_row_axis(seqs[id_]) for id_ in ids_d]).squeeze(axis=1)
    data = X
    return data, ids


def plot_prediction_givenax(model, fig, ntrack, track_no, seq_onehot):
    NUM_CLASSES = model.output_shape[1]
    real_score = model.predict(seq_onehot)[0]
    ax = fig.add_subplot(ntrack, 2, track_no*2-1)
    ax.margins(x=0)
    ax.set_ylabel('Prediction', color='red')
    ax.plot(real_score, '--', color='gray', linewidth=3)
    ax.scatter(range(NUM_CLASSES), real_score, marker='o', color='red', linewidth=11)
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_xticks(range(NUM_CLASSES),)
    ax.set_xticklabels(range(1, NUM_CLASSES+1))
    ax.grid(True)
    return ax


def create_saturation_mutagenesis_x(onehot):
    mutagenesis_X = {"X":[],"ids":[]}
    onehot = onehot.squeeze()
    for mutloc,nt in enumerate(onehot):
        new_X = np.copy(onehot)
        if list(nt) == [1, 0, 0, 0]:
            new_X[mutloc,:] = np.array([0, 1, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 1, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 0, 1], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            mutagenesis_X["ids"].append(str(mutloc)+"_C")
            mutagenesis_X["ids"].append(str(mutloc)+"_G")
            mutagenesis_X["ids"].append(str(mutloc)+"_T")
        if list(nt) == [0, 1, 0, 0]:
            new_X[mutloc,:] = np.array([1, 0, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 1, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 0, 1], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            mutagenesis_X["ids"].append(str(mutloc)+"_A")
            mutagenesis_X["ids"].append(str(mutloc)+"_G")
            mutagenesis_X["ids"].append(str(mutloc)+"_T")
        if list(nt) == [0, 0, 1, 0]:
            new_X[mutloc,:] = np.array([1, 0, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 1, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 0, 1], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            mutagenesis_X["ids"].append(str(mutloc)+"_A")
            mutagenesis_X["ids"].append(str(mutloc)+"_C")
            mutagenesis_X["ids"].append(str(mutloc)+"_T")
        if list(nt) == [0, 0, 0, 1]:
            new_X[mutloc,:] = np.array([1, 0, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 1, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 1, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            mutagenesis_X["ids"].append(str(mutloc)+"_A")
            mutagenesis_X["ids"].append(str(mutloc)+"_C")
            mutagenesis_X["ids"].append(str(mutloc)+"_G")
            
    mutagenesis_X["X"] = np.array(mutagenesis_X["X"])
    return mutagenesis_X 


def plot_mutagenesis_givenax(model, fig, ntrack, track_no, seq_onehot, class_no):
    
    mutagenesis_X = create_saturation_mutagenesis_x(seq_onehot)
    prediction_mutagenesis_X = model.predict(mutagenesis_X["X"])
    original_prediction = model.predict(seq_onehot)
    class_no = class_no-1
    seq_shape = (seq_onehot.shape[1],seq_onehot.shape[2])
    
    arr_a = np.zeros(seq_shape[0])
    arr_c = np.zeros(seq_shape[0])
    arr_g = np.zeros(seq_shape[0])
    arr_t = np.zeros(seq_shape[0])
    delta_pred = original_prediction[:,class_no] - prediction_mutagenesis_X[:,class_no]
    for i,mut in enumerate(mutagenesis_X["ids"]):
        if mut.endswith("A"):
            arr_a[int(mut.split("_")[0])]=delta_pred[i]
        if mut.endswith("C"):
            arr_c[int(mut.split("_")[0])]=delta_pred[i]
        if mut.endswith("G"):
            arr_g[int(mut.split("_")[0])]=delta_pred[i]
        if mut.endswith("T"):
            arr_t[int(mut.split("_")[0])]=delta_pred[i]

    arr_a[arr_a == 0] = None
    arr_c[arr_c == 0] = None
    arr_g[arr_g == 0] = None
    arr_t[arr_t == 0] = None
    
    ax = fig.add_subplot(ntrack, 1, track_no)
    ax.set_ylabel('In silico\nMutagenesis')
    ax.scatter(range(seq_shape[0]), -1*arr_a, label='A', color='green')
    ax.scatter(range(seq_shape[0]), -1*arr_c, label='C', color='blue')
    ax.scatter(range(seq_shape[0]), -1*arr_g, label='G', color='orange')
    ax.scatter(range(seq_shape[0]), -1*arr_t, label='T', color='red')
    ax.legend()
    ax.axhline(y=0, linestyle='--', color='gray')
    ax.set_xlim((0, seq_shape[0]))
    _ = ax.set_xticks(np.arange(0, seq_shape[0]+1, 10))

    return ax


def insilico_evolution(regions, model, class_no, n_mutation):
    #from scipy.stats import zscore
    nuc_to_onehot = {"A":[1, 0, 0, 0],"C":[0, 1, 0, 0],"G":[0, 0, 1, 0],"T":[0, 0, 0, 1]}
    mutation_pred = []
    mutation_loc = []
    print("Sequence index:",end=" ")
    for id_ in range(len(regions)):
        start_x = np.copy(regions[id_:id_+1])
        pred = []
        mut = []
        for i in range(n_mutation):
            mutagenesis_X = create_saturation_mutagenesis_x(start_x)
            prediction_mutagenesis_X = model.predict(mutagenesis_X["X"])
            original_prediction = model.predict(start_x)
            ## To use max z-score
            # next_one = mutagenesis_X["ids"][np.argmax(zscore(prediction_mutagenesis_X-original_prediction,axis=1)[:,class_no-1])]
            ## To use max score
            next_one = mutagenesis_X["ids"][np.argmax(prediction_mutagenesis_X[:,class_no-1]-original_prediction[:,class_no-1])]
            pred.append(original_prediction)
            mut.append(next_one)
            start_x[0][int(next_one.split("_")[0]),:] = np.array(nuc_to_onehot[next_one.split("_")[1]], dtype='int8')
        original_prediction = model.predict(start_x)
        pred.append(original_prediction)
        mutation_pred.append(pred)
        mutation_loc.append(mut)
        print(id_,end=",")
    mutation_pred = np.array(mutation_pred).squeeze()
    mutation_loc = np.array(mutation_loc)
    return mutation_pred, mutation_loc


def random_sequence_by_shuffling(seq_to_shuffle, number_of_random_regions):
    seq_to_shuffle_onehot = one_hot_encode_along_row_axis(seq_to_shuffle)
    shuffled_regions = []
    for i in range(number_of_random_regions):
        np.random.shuffle(seq_to_shuffle_onehot[0])
        shuffled_regions.append(np.copy(seq_to_shuffle_onehot[0]))
    shuffled_regions = np.array(shuffled_regions)
    return shuffled_regions


def random_sequence(seq_len, number_of_random_regions):
    random_regions = []
    for k in range(number_of_random_regions):
        seq = []
        for i in range(seq_len):
            seq.append(np.random.choice(["A","C","G","T"]))
        random_regions.append(one_hot_encode_along_row_axis("".join(seq)).squeeze())
    random_regions = np.array(random_regions)
    return random_regions


def random_sequence_gc_adjusted(seq_len, number_of_random_regions, path_to_use_GC_content):
    regions_to_use_GC = prepare_data(path_to_use_GC_content)
    ACGT_dist = np.sum(regions_to_use_GC[0],axis=0)/len(regions_to_use_GC[0])
    random_regions = []
    for k in range(number_of_random_regions):
        seq = []
        for i in range(seq_len):
            seq.append(np.random.choice(["A","C","G","T"],p=list(ACGT_dist[i])))
        random_regions.append(one_hot_encode_along_row_axis("".join(seq)).squeeze())
    random_regions = np.array(random_regions)
    return random_regions


def plot_deepexplainer_givenax(explainer, fig, ntrack, track_no, seq_onehot):
    shap_values_ = explainer.shap_values(seq_onehot,ranked_outputs=1)
    _, ax1 = plot_weights(shap_values_[0]*seq_onehot,
                          fig, ntrack, 1, track_no,
                          title="", subticks_frequency=10, ylab="")
    return ax1


def load_model(path_json, path_hdf5):
    model_json_file = open(path_json)
    model_json = model_json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(path_hdf5)
    return model


def add_pattern_to_best_location(pattern, regions, model, class_no):
    pattern_added_regions =  np.zeros(regions.shape,dtype="int")
    pattern_locations = np.zeros(regions.shape[0],dtype="int")
    print("Sequence index:",end=" ")
    for r, region in enumerate(regions):
        tmp_array = np.zeros((regions.shape[1]-pattern.shape[1]+1,regions.shape[1],regions.shape[2]))
        for nt in range(tmp_array.shape[0]):
            tmp_array[nt] = np.copy(region)
            tmp_array[nt,nt:nt+pattern.shape[1],:] = pattern[0]
        prediction = model.predict(tmp_array)[:,class_no-1]
        pattern_locations[r] = np.argmax(prediction)    
        pattern_added_regions[r] = tmp_array[pattern_locations[r]]
        print(r,end=",")
    print("")
    return {"regions":pattern_added_regions, "locations":pattern_locations}


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))


default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}


def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if letter[1] > 0:
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    return ax


def plot_weights(array, fig, n, n1, n2, title='', ylab='',
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=20,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    ax = fig.add_subplot(n, n1, n2)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    y = plot_weights_given_ax(ax=ax, array=array,
                              height_padding_factor=height_padding_factor,
                              length_padding=length_padding,
                              subticks_frequency=subticks_frequency,
                              colors=colors,
                              plot_funcs=plot_funcs,
                              highlight=highlight)
    return fig, ax

def random_sequence_dinucleotide_adjusted(seq_len, number_of_random_regions, path_to_use_dinucleotide_content):
    # Step 1: Prepare dinucleotide transition probabilities
    def prepare_dinucleotide_probabilities(path):
        regions = prepare_data(path)  # Your existing function to load the sequences
        sequences = [one_hot_to_sequence(region) for region in regions[0]]
        
        # Count dinucleotide frequencies
        dinucleotide_counts = Counter()
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                dinucleotide = sequence[i:i+2]
                dinucleotide_counts[dinucleotide] += 1
        
        # Normalize to get probabilities
        total_dinucleotide_counts = sum(dinucleotide_counts.values())
        dinucleotide_probs = {key: value / total_dinucleotide_counts for key, value in dinucleotide_counts.items()}
        
        # Build transition probabilities for each nucleotide pair
        transitions = {base: {} for base in "ACGT"}
        for dinucleotide, prob in dinucleotide_probs.items():
            transitions[dinucleotide[0]][dinucleotide[1]] = prob
        
        # Normalize transitions per starting nucleotide
        for base, trans in transitions.items():
            total = sum(trans.values())
            for key in trans:
                trans[key] /= total
        
        return transitions
    
    # Step 2: Generate sequences using the dinucleotide probabilities
    def generate_random_sequence(transitions, seq_len):
        sequence = []
        first_nucleotide = np.random.choice(list("ACGT"))
        sequence.append(first_nucleotide)
        for _ in range(seq_len - 1):
            current_nucleotide = sequence[-1]
            next_nucleotide = np.random.choice(
                list(transitions[current_nucleotide].keys()),
                p=list(transitions[current_nucleotide].values())
            )
            sequence.append(next_nucleotide)
        return "".join(sequence)
    
    # Load and prepare dinucleotide transition probabilities
    transitions = prepare_dinucleotide_probabilities(path_to_use_dinucleotide_content)
    
    # Generate random sequences and one-hot encode them
    random_regions = []
    for _ in range(number_of_random_regions):
        random_seq = generate_random_sequence(transitions, seq_len)
        random_regions.append(one_hot_encode_along_row_axis(random_seq).squeeze())
    
    return np.array(random_regions)

def one_hot_to_sequence(one_hot_encoded):
    nucleotides = "ACGT"
    indices = np.argmax(one_hot_encoded, axis=1)
    return "".join([nucleotides[i] for i in indices])