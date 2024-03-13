
import numpy as np
import pretty_midi
from tqdm import tqdm

from extractor import utils
from extractor.statistics import describe

from extractor.TIS import harmonic_change, compute_dissonance, \
    compute_euclidean_similarity, compute_cosine_similarity
from extractor.divide_score import divide_score_harmony

import warnings
warnings.filterwarnings("ignore")


def midi2chroma(midi_vector):
    # converts piano roll to chromagram
    chroma_vector = np.zeros((midi_vector.shape[0], 12))
    for ii, midi_frame in enumerate(midi_vector):
        for jj, element in enumerate(midi_frame):
            chroma_vector[ii][jj % 12] += element
    return chroma_vector


def hcdf(midi):
    # computes hcdf as defined in Ramoneda 2020
    changes, hcdf_changes, harmonic_function = harmonic_change(chroma=midi, symbolic=True, sigma=26, dist="euclidean")

    return changes


def summarize(piece, ch):
    # summarize the harmony on time boundary
    for ii in range(1, len(ch)):
        s = ch[ii - 1]
        e = ch[ii]
        piece[s:e] = [np.sum(piece[s:e], axis=0)] * (e - s)
    return piece


def compute_descriptor(sing, accomp, type_descriptor):
    # Compute the type_descr for a summarized time serie
    ans = []
    if type_descriptor == 'euclidean similarity':
        ans = compute_euclidean_similarity(midi2chroma(sing), midi2chroma(accomp), True)
    elif type_descriptor == 'cosine similarity':
        ans = compute_cosine_similarity(midi2chroma(sing), midi2chroma(accomp), True)
    return ans


def remove_silences(singer, time_serie):
    new_time_serie = []
    for idx, (sing, t) in enumerate(zip(singer, time_serie)):
        if np.count_nonzero(sing) != 0:
            new_time_serie.append(t)
    return new_time_serie


def count_silences(sing):
    count = 0
    for s in sing:
        if np.count_nonzero(s) != 0:
            count += 1
    return count


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


def extract_descriptor(sing, r_h, l_h, type_descriptor):
    # analyse a track with the type_descr
    sing_pr, accomp_pr = to_chroma(sing, r_h, l_h)
    hcdf_boundaries = hcdf(accomp_pr)

    summarize_accomp = summarize(accomp_pr, hcdf_boundaries)

    descr = compute_descriptor(sing_pr, summarize_accomp, type_descriptor)

    return remove_silences(sing_pr, descr)


def run_extractor(corpus, type_descriptor):
    feature_filename = '../features/harmony' + '_' + corpus + '.json'

    features_dict = {}
    for grade, name, sing, r_h, l_h in tqdm(utils.load_midis(corpus)):
        harmony_time_series = extract_descriptor(sing, r_h, l_h, type_descriptor=type_descriptor)
        features_dict[name] = {
            'time_series': harmony_time_series,
            'statistics': describe(harmony_time_series),
            'grade': grade,
        }

    utils.save_json(features_dict, feature_filename)
    print('Feature file saved as: {}'.format(feature_filename))


if __name__ == "__main__":
    run_extractor(corpus='modern', type_descriptor="cosine similarity")
    run_extractor(corpus='classical', type_descriptor="cosine similarity")
