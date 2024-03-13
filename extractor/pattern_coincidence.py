import pandas as pd
import extractor.utils as ut
import numpy as np


# Function to get feature values for certain task, corpus and pattern size
def get_feature_list(task, corpus, pattern_size, statistic_descriptor='mean'):
    # Put feature filename together
    if task == 'rhythm' or task == 'melody':
        filename = '../features/pattern_coincidence_n/' + task + '_' + corpus + '_' + str(pattern_size) + '.json'
    else:
        filename = '../features/' + task + '_' + corpus + '.json'

    grade_dict = {
        '0': [], '1': [], '2': [], '3': [], '4': [],
        '5': [], '6': [], '7': [], '8': []
    }

    if statistic_descriptor == 'all':
        feature_dict = ut.load_json(filename)
        for p, track in feature_dict.items():
            track['statistics']['name'] = p
            grade_dict[track['grade']].append(track['statistics'])
    else:
        feature_dict = ut.load_json(filename)
        for track in feature_dict.values():
            grade_dict[track['grade']].append(track['statistics'][statistic_descriptor])

    # Getting feature arrays for output
    x = []
    values = []
    for grade, value_list in grade_dict.items():
        x = np.concatenate([x, ['grade ' + grade] * len(value_list)])
        values = np.concatenate([values, value_list])

    return x, values


def pattern_coincidence(task, pattern_sizes=None, corpus='example'):
    df_dict = {}
    for size in pattern_sizes:
        grades, values = get_feature_list(task, corpus, size, 'all')
        df_dict[str(size)] = [value['mean'] for value in values]
    df = pd.DataFrame(df_dict)

    embedding = np.array([np.sum(p) for p in df.to_numpy()]) / pattern_sizes[-1]
    names = [value['name'] for value in values]
    stdevs = np.array([np.std(p) for p in df.to_numpy()])
    pattern_coincidence = {}
    for m, stdev, grade, name in zip(embedding, stdevs, grades, names):
        pattern_coincidence[name] = {
            "grade": str(grade[-1]),
            "statistics": {
                "mean": m,
                "stedev": stdev,
            },
        }
    ut.save_json(pattern_coincidence, "../features/" + task + "_pattern_coincidence_" + corpus + '.json')


if __name__ == '__main__':
    pattern_coincidence('rhythm', [1, 2, 3, 4, 5, 6, 7])
    pattern_coincidence('melody', [1, 2, 3, 4, 5, 6, 7])
