
import os
import numpy as np
from tqdm import tqdm
from music21 import converter

from extractor import utils, statistics
from extractor.onsets_extractor import load_weights
from extractor.statistics import describe

import pretty_midi

import warnings
warnings.filterwarnings("ignore")


def compute_descriptor(score_path):
    # Get MIDI score path for three parts
    sing_part_path = score_path
    right_hand_part_path = score_path.replace('sing.mid', 'rh.mid')
    left_hand_part_path = score_path.replace('sing.mid', 'lh.mid')

    # Load MIDI scores using pretty_midi
    sing_part = pretty_midi.PrettyMIDI(sing_part_path)
    right_hand_part = pretty_midi.PrettyMIDI(right_hand_part_path)
    left_hand_part = pretty_midi.PrettyMIDI(left_hand_part_path)

    # Load instrument part from the pretty_midi objects
    voice = sing_part.instruments[0]
    right_hand = right_hand_part.instruments[0]
    left_hand = left_hand_part.instruments[0]

    # Get onsets from parts
    voice_onsets = []
    for nt in voice.notes:
        voice_onsets.append(nt.start)
    right_hand_onsets = []
    for nt in right_hand.notes:
        right_hand_onsets.append(nt.start)
    left_hand_onsets = []
    for nt in left_hand.notes:
        left_hand_onsets.append(nt.start)

    # Sort onsets
    voice_onsets = sorted(voice_onsets)
    right_hand_onsets = sorted(right_hand_onsets)
    left_hand_onsets = sorted(left_hand_onsets)
    
    rhythm_doubled = []
    for bar in np.arange(len(voice_onsets)):
        # Get respective onsets from vocal onsets (if any)
        tmp_voice = voice_onsets[bar]
        sr_ind = right_hand_onsets.index(tmp_voice) if tmp_voice in right_hand_onsets else None
        sl_ind = left_hand_onsets.index(tmp_voice) if tmp_voice in left_hand_onsets else None

        if sr_ind is None and sl_ind is None:  # No respective onset on vocal onset
            rhythm_doubled.append(0)
        else:  # Respective onset at right or left hand
            rhythm_doubled.append(1)

    # Load metrical hierarchy weights from mat files
    weights = load_weights(score_path)

    # Compute and return final array
    return np.multiply(np.array(rhythm_doubled), weights/5)


def run_extractor(corpus):

    feature_filename = '../features/onset' + '_' + corpus + '.json'
    grade_map_filename = '../voice_and_co/' + corpus + '.json'

    # Get tracks for each grade
    grade_map = utils.load_json(grade_map_filename)

    # Iterate over tracks in grades to compute the RO
    features_dict = {}
    for grade, track_list in tqdm(grade_map.items()):
        for track in track_list:
            sing_midi_path = track + '.sib_sing.mid'

            rhythm_doubled_time_series = compute_descriptor(
                sing_midi_path
            )
            features_dict[track] = {
                'statistics': describe(rhythm_doubled_time_series),
                'grade': grade,
            }

    # Store feature dict for visualization
    utils.save_json(features_dict, feature_filename)
    print('Feature file saved as: {}'.format(feature_filename))


if __name__ == "__main__":
    run_extractor(corpus='classical')
    run_extractor(corpus='modern')
