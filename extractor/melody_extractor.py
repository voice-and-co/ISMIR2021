

import numpy as np
from pretty_midi import pretty_midi
from tqdm import tqdm
from itertools import groupby

from extractor.divide_score import divide_score_melody
from extractor.statistics import describe
from extractor import utils

import warnings
warnings.filterwarnings("ignore")


def add_chroma_zeros(part, max_size):
    # Padding function for chroma matrices
    if part.shape[0] != max_size:
        return np.concatenate([part, np.zeros((max_size - part.shape[0], 12))], axis=0)
    else:
        return part


def to_chroma(sing, r_h, l_h):
    # Load MIDI score parts in chroma (voice, right hand, left hand) using pretty_midi
    chroma_sing = pretty_midi.PrettyMIDI(sing).get_chroma().transpose()
    chroma_r_h = pretty_midi.PrettyMIDI(r_h).get_chroma().transpose()
    chroma_l_h = pretty_midi.PrettyMIDI(l_h).get_chroma().transpose()

    # Compute max size of chroma and return
    max_size = max(chroma_sing.shape[0], chroma_r_h.shape[0], chroma_l_h.shape[0])
    chroma_sing = add_chroma_zeros(chroma_sing, max_size)
    chroma_r_h = add_chroma_zeros(chroma_r_h, max_size)
    chroma_l_h = add_chroma_zeros(chroma_l_h, max_size)
    return chroma_sing, chroma_r_h + chroma_l_h


def to_binary(midi_chroma):
    # Set to 1 if sound is present
    for i in np.arange(np.shape(midi_chroma)[0]):
        for j in np.arange(np.shape(midi_chroma)[1]):
            if midi_chroma[i][j] > 0:
                midi_chroma[i][j] = 1
    return midi_chroma


def compute_descriptor(sing_path, r_h_path, l_h_path, pattern_size):
    # Convert lead melody and accompaniment hands to padded chroma
    sing, accomp = to_chroma(sing_path, r_h_path, l_h_path)

    # Convert chroma matrices to binary
    melody = to_binary(sing)
    accomp = to_binary(accomp)

    # Get melody patterns for pattern_size 1
    melody_patterns = [
        (k, list(g)) for k, g in groupby([np.argmax(m) if np.count_nonzero(m) != 0 else -1 for m in melody])
    ]
    start = 0
    doubled = []
    # Run across the parsed segments and compute coincidence
    for k, melody_segment in melody_patterns:
        frames_note = len(melody_segment)
        accomp_segment = accomp[start:start + frames_note][:]

        count_doubled = 0
        for m, a in zip(melody_segment, accomp_segment):
            if m == -1 or a[m] != 0:
                count_doubled += 1

        if count_doubled == frames_note:
            doubled.append((True, frames_note))
        else:
            doubled.append((False, frames_note))
        start += frames_note

    time_serie = []
    # Extend the algorithm to pattern size > 1
    for start in range(len(doubled) - pattern_size):
        segment = doubled[start:start + pattern_size]
        segment_doubled, segment_weights = zip(*segment)
        weights = np.sum(segment_weights)
        if np.all(segment_doubled):
            time_serie = time_serie + [1 for _ in range(weights)]
        else:
            time_serie = time_serie + [0 for _ in range(weights)]
    return time_serie


def run_extractor(corpus, pattern_size):

    feature_filename = '../features/pattern_coincidence_n/melody' + '_' + corpus + '_' + str(pattern_size) + '.json'

    # Compute descriptor for certain corpus and pattern size
    features_dict = {}
    for grade, name, sing, r_h, l_h in tqdm(utils.load_midis(corpus)):
        melody_pattern_coincidence = compute_descriptor(sing, r_h, l_h, pattern_size)
        features_dict[name] = {
            'statistics': describe(melody_pattern_coincidence),
            'grade': grade,
        }

    # Save feature dictionary for visualization
    utils.save_json(features_dict, feature_filename)
    print('Feature file saved as: {}'.format(feature_filename))
