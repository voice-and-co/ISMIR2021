"""
    File name: visualize_results.py
    Author: Genis Plaja and Pedro Ramoneda
    Date created: 4/17/2020
    Python Version: 3.8
    Description: plotting library to visualize the results
"""
import os
import pprint

import extractor.utils as ut
import plotly.graph_objects as go
import numpy as np


def create_ids():
    # create ids for each music piece in the collections
    parse_dict = {}
    ii = 0
    for corpus in ['classical', 'modern']:
        path = '../voice_and_co/' + corpus + '.json'
        data = ut.load_json(path)
        for grade, paths in data.items():
            for p in paths:
                parse_dict[p] = ii
                ii += 1
    ut.save_json(parse_dict, '../voice_and_co/to_id.json')
    ut.save_json({k: p for p, k in parse_dict.items()}, '../voice_and_co/to_path.json')


cache_distributions = {}


def create_text_hover(corpus, values, sizes=None):
    # return boxcar hover for a Track
    to_id = ut.load_json('../voice_and_co/to_id.json')
    texts = []
    for value in values:
        summ = get_track_summary(corpus, value['name'])
        text = "<br>"
        text = text + '<b>NAME' + ':</b> ' + value['name'].split('/')[-1] + '<br>'
        text = text + '<b>ID' + ':</b> ' + str(to_id[value['name']]) + '<br>'
        text = text + '<br>'
        text = text + 'MEAN: ' + str(round(value['mean'], 2)) + '<br>'
        if 'stdev' in value:
            text = text + 'STDEV: ' + str(round(value['stdev'], 2)) + '<br>'
        text = text + ' <br>'

        features = {}
        for feat, val in summ.items():
            if not (('melody' in feat or 'rhythm' in feat) and feat[-1] in ['8', '9']):
                features[feat] = val

        rhythm_feat = [str(round(features[feat]['mean'], 2)) for feat in sorted(features.keys()) if 'rhythm' in feat]
        melody_feat = [str(round(features[feat]['mean'], 2)) for feat in sorted(features.keys()) if 'melody' in feat]
        harmony_feat = [str(round(features[feat]['mean'], 2)) for feat in sorted(features.keys()) if 'harmony' in feat]
        onset_feat = [str(round(features[feat]['mean'], 2)) for feat in sorted(features.keys()) if 'onset' in feat]
        contour_feat = [str(round(features[feat]['mean'], 2)) for feat in sorted(features.keys()) if 'contour' in feat]

        text = text + "RO: " + ', '.join(onset_feat) + '<br>'
        text = text + "RPC: " + ', '.join(rhythm_feat) + '<br>'
        text = text + "MC: " + ', '.join(contour_feat) + '<br>'
        text = text + "MPC: " + ', '.join(melody_feat) + '<br>'
        text = text + "H: " + ', '.join(harmony_feat) + '<br>'

        texts.append(text)
    return texts


def plot_features(task, pattern_sizes=None):
    # plot and save interactive boxcar plots
    if not os.path.exists('../output_figs/'):
        os.mkdir('../output_figs/')

    fig = go.Figure()

    if task == 'rhythm' or task == 'melody':
        corpora = ['classical', 'modern']
        for corpus in corpora:
            for size in pattern_sizes:
                x, values = get_feature_list(task, corpus, size, 'all')
                fig.add_trace(go.Box(
                    x=x,
                    y=[value['mean'] for value in values],
                    name=('Classical' if corpus == 'classical' else 'Rock & Pop') + ' - Size: ' + str(size),
                    # boxmean='sd',
                    boxmean=True,
                    boxpoints='all',
                    jitter=0.2,
                    hoverinfo='text',
                    text=create_text_hover(corpus, values, pattern_sizes)
                ))
    else:
        corpora = ['classical', 'modern']
        for corpus in corpora:
            x, values = get_feature_list(task, corpus, 0, 'all')
            fig.add_trace(go.Box(
                x=x,
                y=[value['mean'] for value in values],
                name='Classical' if corpus == 'classical' else 'Rock & Pop',
                # boxmean='sd',
                boxmean=True,
                boxpoints='all',
                jitter=0.2,
                hoverinfo='text',
                text=create_text_hover(corpus, values, pattern_sizes)
            ))

    fig.update_layout(
        hoverlabel=dict(
            font_size=26,
            font_family="Rockwell"
        )
    )

    fig.update_layout(
        title='ANALYSIS OF THE ' + task.upper().replace('_', ' ') + ' RELATIONSHIP BETWEEN VOCAL LINE AND ' +
              'ACCOMPANIMENT ALONG THE CORPUS GRADES',
        xaxis_title='GRADES',
        yaxis_title=task.upper() + ' DESCRIPTOR',
        boxmode='group'  # group together boxes of the different traces for each value of x
    )

    fig.write_html("../output_figs/" + task + '.html')
    fig.show()


def plot_features_paper(task, pattern_sizes=None):
    # plot and save paper pdf plots
    if not os.path.exists('../output_figs/'):
        os.mkdir('../output_figs/')

    fig = go.Figure()

    if task == 'melody' or task == 'rhythm':
        corpora = ['classical', 'modern']
        for corpus in corpora:
            for size in pattern_sizes:
                x, values = get_feature_list(task, corpus, size, 'all')
                fig.add_trace(go.Box(
                    x=x,
                    y=[value['mean'] for value in values],
                    # boxpoints=False,
                    name=('Classical' if corpus == 'classical' else 'Rock & Pop') + ' - Size: ' + str(size),
                    boxmean=True,
                ))
    else:
        corpora = ['classical', 'modern']
        for corpus in corpora:
            x, values = get_feature_list(task, corpus, 0, 'all')
            fig.add_trace(go.Box(
                x=x,
                y=[value['mean'] for value in values],
                # boxpoints=False,
                name='Classical' if corpus == 'classical' else 'Rock & Pop',
                boxmean=True,
            ))

    fig.update_layout(
        hoverlabel=dict(
            font_size=26,
            font_family="Rockwell"
        )
    )

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ))
    if task == 'rhythm_pattern_coincidence' or task == 'melody_pattern_coincidence':
        title = task.replace('_', ' ').upper() + ' DESCRIPTOR'
    elif task == 'onset':
        title = 'RHYTHM ' + task.upper() + ' DESCRIPTOR'
    elif task == 'contour':
        title = 'MELODY ' + task.upper() + ' DESCRIPTOR'
    else:
        title = task.upper() + ' DESCRIPTOR'
    fig.update_layout(
        # title=task.upper() + ' RELATIONSHIP',
        xaxis_title='GRADES',
        yaxis_title=title,
        boxmode='group'  # group together boxes of the different traces for each value of x
    )

    fig.write_image("../output_figs/" + task + '.pdf')
    fig.show(renderer='pdf')


def get_feature_list(task, corpus, pattern_size, statistic_descriptor='mean'):
    # get all the features of a descriptor [task] a corpus [classical|rock&pop] and a pattern size [patternsize]
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

    x = []
    values = []
    for grade, value_list in grade_dict.items():
        x = np.concatenate([x, ['grade ' + grade] * len(value_list)])
        values = np.concatenate([values, value_list])

    return x, values


def get_all_features(corpus):
    # get all the features for a table in a corpus
    dimensions = ['harmony', 'melody_pattern_coincidence', 'contour', 'onset', 'rhythm_pattern_coincidence']
    descriptors = ["mean"]
    X, y, X_labels, y_labels = [], [], [], []
    for dimension in dimensions:
        for descriptor in descriptors:
            y_labels, feature_list = get_feature_list(dimension, corpus, 0, descriptor)
            X.append(feature_list)
            label = ""
            if "onset" == dimension:
                label = "RO"
            elif "melody_pattern_coincidence" == dimension:
                label = "MPC"
            elif "rhythm_pattern_coincidence" == dimension:
                label = "RPC"
            elif "contour" == dimension:
                label = "MC"
            elif "harmony" == dimension:
                label = "H"
            X_labels.append(label)

    y = [int(label[-1]) for label in y_labels]
    return np.array(X).transpose(), y, X_labels, y_labels


def get_track_list():
    # get the list of all the tracks
    classical_path = '../voice_and_co/classical.json'
    modern_path = '../voice_and_co/modern.json'

    classical_dict = ut.load_json(classical_path)
    modern_dict = ut.load_json(modern_path)

    print('Listing classical tracks...')
    pprint.pprint(classical_dict)
    print('Visualizing modern dict...')
    pprint.pprint(modern_dict)


summary_cache = {}
def get_track_summary(corpus, track_name):
    # get all the track descriptors
    track_summary = {}
    for feature_file in os.listdir('../features/'):
        if corpus in feature_file:
            if feature_file in summary_cache:
                features = summary_cache[feature_file]
            else:
                features = ut.load_json(os.path.join('../features', feature_file))
                all_mean = [t['statistics']['mean'] for k, t in features.items()]
                minim = min(all_mean)
                maxim = max(all_mean)
                for k, p in zip(features.keys(), all_mean):
                    features[k]['statistics']['percentile'] = (p - minim) / (maxim - minim)

                summary_cache[feature_file] = features
            track_features = features[track_name]
            track_summary[feature_file.replace('.json', '').replace('midi_', '')] = track_features['statistics']
    return track_summary


if __name__ == '__main__':
    plot_features_paper('onset')
    plot_features_paper('rhythm_pattern_coincidence')
    plot_features_paper('contour')
    plot_features_paper('melody_pattern_coincidence')
    plot_features_paper('harmony')


