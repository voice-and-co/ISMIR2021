
import json
import os


def load_midis(corpus):
    if corpus == 'modern':
        path_corpus = '../voice_and_co/modern.json'
    elif corpus == 'classical':
        path_corpus = '../voice_and_co/classical.json'
    else:
        path_corpus = '../voice_and_co/example.json'
    grade, name, voice, r_h, l_h = [], [], [], [], []
    for g, paths in load_json(path_corpus).items():
        for path in paths:
            grade.append(g)
            name.append(path)
            voice.append(path + '.sib_sing.mid')
            r_h.append(path + '.sib_rh.mid')
            l_h.append(path + '.sib_lh.mid')

    return zip(grade, name, voice, r_h, l_h)


def save_json(dictionary, name_file):
    with open(name_file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def load_json(name_file):
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data
