import os
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns
from pybedtools import BedTool
import numpy as np

    
class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
    """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""
    def _open(file):
        return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)
    
    with _open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            chrom, pos, id, ref, alt_list = line.split('\t')[:5]
            # Split ALT alleles and return individual variants as output.
            for alt in alt_list.split(','):
                yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                                   ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=FastaStringExtractor(fasta_file))

    for variant in variant_generator(vcf_file, gzipped=gzipped):
        interval = Interval(chr_prefix + variant.chrom,
                            variant.pos, variant.pos)
        interval = interval.resize(sequence_length)
        center = interval.center() - interval.start

        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)

        yield {'inputs': {'ref': one_hot_encode(reference),
                          'alt': one_hot_encode(alternate)},
               'metadata': {'chrom': chr_prefix + variant.chrom,
                            'pos': variant.pos,
                            'id': variant.id,
                            'ref': variant.ref,
                            'alt': variant.alt}}
        
# @title `Enformer`, `EnformerScoreVariantsNormalized`, `EnformerScoreVariantsPCANormalized`,
SEQUENCE_LENGTH = 393216

class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                  target_mask, output_head='mouse'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(
              target_mask[tf.newaxis] *
              self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)


class EnformerScoreVariantsRaw:

    def __init__(self, tfhub_url, organism='human'):
        self._model = Enformer(tfhub_url)
        self._organism = organism

    def predict_on_batch(self, inputs):
        ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
        alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

        return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
               organism='human'):
        assert organism == 'human', 'Transforms only compatible with organism=human'
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            transform_pipeline = joblib.load(f)
        self._transform = transform_pipeline.steps[0][1]  # StandardScaler.

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
               organism='human', num_top_features=500):
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            self._transform = joblib.load(f)
        self._num_top_features = num_top_features

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)[:, :self._num_top_features]


def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()

    
# @title Compute contribution scores
def get_contribution_scores(model, seq, track_nr, seq_len=500):
    predictions = model.predict_on_batch(seq[np.newaxis])['mouse'][0]

    target_mask = np.zeros_like(predictions)
    for idx in [447, 448, 449]:
        target_mask[idx, track_nr] = 1
        #target_mask[idx, 183] = 1
    # This will take some time since tf.function needs to get compiled.
    contribution_scores = model.contribution_input_grad(seq.astype(np.float32), target_mask).numpy()
    contribution_scores = np.repeat(contribution_scores[196608-int(seq_len/2):196608+int(seq_len/2)],4)
    contribution_scores = np.reshape(contribution_scores, (500,4))
    return contribution_scores
