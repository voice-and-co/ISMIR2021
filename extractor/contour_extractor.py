

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
    # padding function
    if part.shape[0] != max_size:
        return np.concatenate([part, np.zeros((max_size - part.shape[0], 12))], axis=0)
    else:
        return part


def to_chroma(sing, r_h, l_h):
    chroma_sing = pretty_midi.PrettyMIDI(sing).get_chroma().transpose()
    chroma_r_h = pretty_midi.PrettyMIDI(r_h).get_chroma().transpose()
    chroma_l_h = pretty_midi.PrettyMIDI(l_h).get_chroma().transpose()

    max_size = max(chroma_sing.shape[0], chroma_r_h.shape[0], chroma_l_h.shape[0])
    chroma_sing = add_chroma_zeros(chroma_sing, max_size)
    chroma_r_h = add_chroma_zeros(chroma_r_h, max_size)
    chroma_l_h = add_chroma_zeros(chroma_l_h, max_size)
    return chroma_sing, chroma_r_h + chroma_l_h


def to_binary(midi_chroma):
    # set to 1 if sound is present
    for i in np.arange(np.shape(midi_chroma)[0]):
        for j in np.arange(np.shape(midi_chroma)[1]):
            if midi_chroma[i][j] > 0:
                midi_chroma[i][j] = 1
    return midi_chroma

def compute_not_doubled(melody, accomp):
    # calculate not doubled melody
    diff = melody - accomp
    count = np.sum(diff[diff > 0])
    return count


def count_silences(melody):
    melody_silences = 0
    for m in melody:
        if sum(m) == 0:
            melody_silences += 1
    return melody_silences


def compute_descriptor(sing_path, r_h_path, l_h_path):
    sing, accomp = to_chroma(sing_path, r_h_path, l_h_path)

    # Compute the type_descr for a summarized time series
    melody = to_binary(sing)
    accomp = to_binary(accomp)
    # melody_silences = count_silences(melody)

    melody_contour = []
    for m, a in zip(melody, accomp):
        if sum(m) != 0:
            if a[m.argmax()] != 0:
                melody_contour.append(1)
            else:
                melody_contour.append(0)

    return melody_contour


def run_extractor(corpus):
    feature_filename = '../features/contour' + '_' + corpus + '.json'

    features_dict = {}
    for grade, name, sing, r_h, l_h in tqdm(utils.load_midis(corpus)):
        contour = compute_descriptor(sing, r_h, l_h)
        features_dict[name] = {
            'statistics': describe(contour),
            'grade': grade,
        }

    utils.save_json(features_dict, feature_filename)
    print('Feature file saved as: {}'.format(feature_filename))


if __name__ == "__main__":
    run_extractor(corpus='modern')
    run_extractor(corpus='classical')
