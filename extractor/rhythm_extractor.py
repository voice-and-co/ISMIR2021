
import os
from operator import attrgetter

import numpy as np
from tqdm import tqdm
from music21 import converter

from extractor import utils, statistics
from extractor.statistics import describe

import pretty_midi

import warnings
warnings.filterwarnings("ignore")


def compute_descriptor(score_path, pattern_size):
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
    
    # Sort notes by start time to match with the onsets
    sorted_voice_notes = sorted(voice.notes, key=attrgetter('start'))
    sorted_right_hand = sorted(right_hand.notes, key=attrgetter('start'))
    sorted_left_hand = sorted(left_hand.notes, key=attrgetter('start'))
    
    rhythm_doubled = []
    for bar in np.arange(len(voice_onsets)):
        # Get respective onsets from vocal onsets (if any)
        tmp_voice = voice_onsets[bar]
        sr_ind = right_hand_onsets.index(tmp_voice) if tmp_voice in right_hand_onsets else None
        sl_ind = left_hand_onsets.index(tmp_voice) if tmp_voice in left_hand_onsets else None

        # No accompaniment onset for the vocal onset
        if sr_ind is None and sl_ind is None:
            rhythm_doubled.append(0)

        # Onset in right hand for vocal onset
        elif sr_ind is not None and sl_ind is None:
            # Make sure duration is the same for notes in onset
            dur_voice = float(sorted_voice_notes[bar].get_duration())
            dur_right = float(sorted_right_hand[sr_ind].get_duration())
            if dur_voice == dur_right:
                rhythm_doubled.append(1)
            else:
                rhythm_doubled.append(0)

        # Onset in left hand for vocal onset
        elif sl_ind is not None and sr_ind is None:
            # Make sure duration is the same for notes in onset
            dur_voice = float(sorted_voice_notes[bar].get_duration())
            dur_left = float(sorted_left_hand[sl_ind].get_duration())
            if dur_voice == dur_left:
                rhythm_doubled.append(1)
            else:
                rhythm_doubled.append(0)

        # Onset in both hands for vocal onset
        elif sl_ind is not None and sr_ind is not None:
            # Make sure duration is the same for all notes in onset
            dur_voice = float(sorted_voice_notes[bar].get_duration())
            dur_right = float(sorted_right_hand[sr_ind].get_duration())
            dur_left = float(sorted_left_hand[sl_ind].get_duration())
            if dur_voice == dur_right or dur_voice == dur_left:
                rhythm_doubled.append(1)
            else:
                rhythm_doubled.append(0)

    time_serie = []
    # Extend function to pattern_size > 1
    for start in range(len(rhythm_doubled) - pattern_size + 1):
        segment = rhythm_doubled[start:start + pattern_size]
        if np.all(segment):
            time_serie.append(1)
        else:
            time_serie.append(0)

    return time_serie


def run_extractor(corpus, pattern_size):

    feature_filename = '../features/pattern_coincidence_n/rhythm' + '_' + corpus + '_' + str(pattern_size) + '.json'
    grade_map_filename = '../voice_and_co/' + corpus + '.json'

    # Get tracks for each grade
    grade_map = utils.load_json(grade_map_filename)

    # Compute descriptor for certain corpus and pattern size
    features_dict = {}
    for grade, track_list in tqdm(grade_map.items()):
        for track in track_list:
            sing_midi_path = track + '.sib_sing.mid'
            rhythm_doubled_time_series = compute_descriptor(
                sing_midi_path,
                pattern_size
            )
            features_dict[track] = {
                'statistics': describe(rhythm_doubled_time_series),
                'grade': grade,
            }

    # Save feature dict for visualization
    utils.save_json(features_dict, feature_filename)
    print('Feature file saved as: {}'.format(feature_filename))
