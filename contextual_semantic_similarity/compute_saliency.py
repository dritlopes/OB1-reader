import pandas as pd
import numpy as np
from collections import defaultdict
import math

def split_data(eye_data, words_data, split=.7):

    all_text_ids = words_data['trialid'].unique().tolist()
    index = int(len(all_text_ids)*split)
    val_text_ids = all_text_ids[:index]
    test_text_ids = all_text_ids[index:]

    eye_data_copy = eye_data.copy()
    words_data_copy = words_data.copy()

    # split eye-tracking and word data based on text ids
    val_eye_data = eye_data_copy[eye_data['trialid'].isin(val_text_ids)]
    val_words_data = words_data_copy[words_data['trialid'].isin(val_text_ids)]
    test_eye_data = eye_data_copy[eye_data['trialid'].isin(test_text_ids)]
    test_words_data = words_data_copy[words_data['trialid'].isin(test_text_ids)]

    return val_eye_data, val_words_data, test_eye_data, test_words_data

def compute_letter_map(words_df):

    # map each word in each text to its letter indices at the text level
    letter_map = defaultdict(list)

    for i, text in words_df.groupby('trialid'):
        position_in_text = 0
        for row in text.itertuples():
            all_letter_ids = [i + 1 for i, character in enumerate(row.texts)]
            letter_ids = all_letter_ids[position_in_text:position_in_text + row.length]
            position_in_text += row.length + 1
            letter_map[f'{int(row.trialid)}-{int(row.ianum)}'] = letter_ids

    return letter_map

def find_letter_distance_2centre_of_context_word(context_word, letter_map, centre_next_word = None):

    # TODO compute centre_next_word
    # compute distance in letters from fixated letter to the centre of context word
    all_let_pos_context_word = letter_map[f'{context_word.trialid}-{context_word.context_ianum}']
    if all_let_pos_context_word:
        centre_let_pos_context_word = all_let_pos_context_word[math.ceil(len(all_let_pos_context_word) / 2) - 1]
        if centre_next_word: # reference point is the centre of next word (to skew attention to n+1)
            let_pos_fix_word = centre_next_word
        else: # reference point is letter being fixated
            let_pos_fix_word = context_word.letternum
        letter_distance = centre_let_pos_context_word - let_pos_fix_word + 1 # + 1 to skew centre to the right
    else:
        raise ValueError(f'{context_word.trialid},{context_word.context_ianum} is empty in letter map \n {context_word}')

    return letter_distance

def baseline_7letter_2right(context, letter_map):

    # word predicted is 7 letters to the right of the fixated letter,
    # and logically the letter predicted is 7 letters to the right of fixated letter

    # optimal saccade distance visual field
    distance_letter7_2right = 7.0
    # which word is 7 letters to the right of fixated letter
    word_letter7_2right = None
    # position of letter being fixated
    fix_letter = context['letternum'].tolist()[0]
    # find position of letter 7 letter positions to the right of the letter being fixated
    letter7_2right = fix_letter + distance_letter7_2right

    for context_word in context.itertuples():
        # find letter positions of context word
        let_positions = letter_map[f'{context_word.trialid}-{context_word.context_ianum}']
        # check if the letter position 7 letters to the right is in this context word
        if letter7_2right in let_positions:
            word_letter7_2right = context_word.distance

    return word_letter7_2right, distance_letter7_2right

def winner_takes_all(context, letter_map, feature, condition, weight_type='raw'):

    winner_score, winner, winner_pos, winner_let_pos = None, None, None, None
    distance_weights_max = {-3:.25, -2:.50, -1:.75, 0:1, 1:1, 2:.75, 3:.5}
    distance_weights_min = {-3:1, -2:.75, -1:.50, 0:.25, 1:.25, 2:.50, 3:.75}

    # print('Fix word: ', context['ia'].tolist()[0])
    for i, context_word in context.iterrows():
        score = context_word[f'{feature}']
        # print('Winner score: ', winner_score)
        # print('Context word', context_word.context_ia)
        # print('Context word position', context_word.distance)
        # print('Score:', score)
        if weight_type == 'distance':
            if condition == 'max':
                score = score*distance_weights_max[context_word.distance]
            elif condition == 'min':
                score = score*distance_weights_min[context_word.distance]
        if not winner_score:
            winner_score = score
        if condition == "max":
            if score >= winner_score:
                winner_score = score
                winner = context_word
        elif condition == "min":
            if score <= winner_score:
                winner_score = score
                winner = context_word

    if winner is not None:
        winner_pos = winner.distance
        winner_let_pos = find_letter_distance_2centre_of_context_word(winner, letter_map)
    # print('Winner: ', winner)
    # print('Winner position: ', winner_pos, winner_let_pos)

    return winner_pos, winner_let_pos

def centre_of_mass(saliencies, positions):

    # print(saliencies)
    # print(positions)
    # print([saliency*position for saliency,position in zip(saliencies,positions)])
    # print(1/np.sum(saliencies))

    return (1/np.sum(saliencies))*np.sum([saliency*position for saliency,position in zip(saliencies,positions)])

def normalize(all_values):

    norm_feature = []

    min_feature = min([x for x in all_values if not math.isnan(x)])
    max_feature = max([x for x in all_values if not math.isnan(x)])

    for x in all_values:
        norm = None
        if x:
            norm = (x - min_feature) / (max_feature - min_feature)
        norm_feature.append(norm)

    return norm_feature

def compute_combi_len_ss(context, ep=1., weight_type='raw'):

    scores = []
    distance_weights = {-3: .25, -2: .50, -1: .75, 0: .75, 1: 1, 2: .75, 3: .5}

    for context_word in context.itertuples():

        features = {'similarity': context_word.similarity,
                    'entropy': context_word.norm_entropy,
                    'length': context_word.length,
                    'surprisal': context_word.surprisal}

        # to account for normalization starting from 0
        for k,v in features.items():
            if v == 0.0:
                features[k] = .0001

        # compute combi
        if not np.isnan(list(features.values())).any():
            score = (1 / (features['similarity'] / features['length'])) + (features['entropy']*ep) + features['surprisal']
            # weight by distance
            if weight_type == 'distance':
                score = score * distance_weights[context_word.distance]
        else:
            score = np.nan
        scores.append(score)

    return scores

def compute_saliency(df, words_df, filepath):

    saliency_types = ['next_word',
                      '7letter_2right',
                      'max_length',
                      'min_frequency',
                      'max_surprisal',
                      'min_semantic_similarity',
                      'mass_length',
                      'mass_frequency',
                      'mass_surprisal',
                      'mass_semantic_similarity',
                      'dist_max_length',
                      'dist_min_frequency',
                      'dist_max_surprisal',
                      'dist_min_semantic_similarity',
                      'combi_len_sur_en_ss',
                      'combi_mass_len_sur_en_ss',
                      'combi_dist_len_sur_en_ss']

    saliency_variables = {'participant_id': [],
                          'text_id': [],
                          'context_window': [],
                          'start_ia': [],
                          'start_position' : [],
                          'end_ia': [],
                          'end_position' : [],
                          'end_relative_position': [],
                          'end_letter_relative_position': [],
                          'saliency_type': [],
                          'pred_end_relative_position': [],
                          'pred_end_letter_relative_position': []}

    # find index of all letters in each word in each text
    letter_map = compute_letter_map(words_df)

    # normalize variables for combination computation
    for feature in ['length', 'entropy', 'surprisal']:
        norm_feature = normalize(df[feature].tolist())
        df[f'norm_{feature}'] = norm_feature
    # convert entropy values from previous context to 1
    df['norm_entropy'] = df.apply(lambda x: 1. if x['distance'] in [-3, -2, -1, 0] else x['norm_entropy'], axis = 1)

    # iter through each fixation
    for i, context in df.groupby(['participant_id','trialid','fixid']):

        for variable in saliency_types:

            saliency_variables['participant_id'].append(i[0])
            saliency_variables['text_id'].append(i[1])
            saliency_variables['context_window'].append(' '.join(context['context_ia'].tolist()))
            saliency_variables['start_ia'].append(context['ia'].tolist()[0])
            saliency_variables['start_position'].append(context['ianum'].tolist()[0])
            saliency_variables['saliency_type'].append(variable)

            # register true landing position
            for context_word in context.itertuples():

                if context_word.landing_target:
                    saliency_variables['end_position'].append(context_word.context_ianum)
                    saliency_variables['end_ia'].append(context_word.context_ia)
                    saliency_variables['end_relative_position'].append(context_word.distance)
                    saliency_variables['end_letter_relative_position'].append(context_word.letter_distance)

            # compute end relative position using saliency
            pred_end_relative_position, pred_end_letter_relative_position = None, None

            # baseline n+1
            if variable == 'next_word':
                pred_end_relative_position = 1
                letter_distance = None
                for context_word in context.itertuples():
                    if context_word.distance == 1:
                        letter_distance = find_letter_distance_2centre_of_context_word(context_word, letter_map)
                pred_end_letter_relative_position = letter_distance

            # baseline 7 letters to right of center of fixation
            elif variable == '7letter_2right':
                pred_end_relative_position, pred_end_letter_relative_position = baseline_7letter_2right(context, letter_map)

            # longest word wins
            elif variable == 'max_length':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context, letter_map,
                                                                                                feature ='length',
                                                                                                condition="max")
            # longest word weighted by distance
            elif variable == 'dist_max_length':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context, letter_map,
                                                                                                  feature='length',
                                                                                                  condition="max",
                                                                                                  weight_type='distance')
            # less frequent word wins
            elif variable == 'min_frequency':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context,letter_map,
                                                                                                feature ='frequency',
                                                                                                condition="min")
            # less frequent weighted by distance
            elif variable == 'dist_min_frequency':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context, letter_map,
                                                                            feature='frequency',
                                                                            condition="min",
                                                                            weight_type='distance')
            # more surprising word wins
            elif variable == 'max_surprisal':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context,letter_map,
                                                                                                feature ='surprisal',
                                                                                                condition="max")
            # more surprising weighted by distance
            elif variable == 'dist_max_surprisal':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context, letter_map,
                                                                                                feature='surprisal',
                                                                                                condition="max",
                                                                                                weight_type='distance')
            # less similar word wins
            elif variable == 'min_semantic_similarity':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context,letter_map,
                                                                                                 feature ='similarity',
                                                                                                 condition="min")
            # less similar weighted by distance
            elif variable == 'dist_min_semantic_similarity':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context,letter_map,
                                                                                                 feature ='similarity',
                                                                                                 condition="min",
                                                                                                 weight_type='distance')

            # centre of mass
            elif variable == 'mass_length':
                pred_end_relative_position = centre_of_mass(context['length'].tolist(), context['distance'].tolist())
            elif variable == 'mass_frequency':
                pred_end_relative_position = centre_of_mass(context['frequency'].tolist(), context['distance'].tolist())
            elif variable == 'mass_surprisal':
                pred_end_relative_position = centre_of_mass(context['surprisal'].tolist(), context['distance'].tolist())
            elif variable == 'mass_semantic_similarity':
                pred_end_relative_position = centre_of_mass(context['similarity'].tolist(), context['distance'].tolist())

            # combination of length, surprisal, entropy and semantic similarity
            elif variable == 'combi_len_sur_en_ss':
                pred_end_relative_position, pred_end_letter_relative_position = None, None
                scores = compute_combi_len_ss(context, ep=.5)
                if not np.isnan(scores).any(): # only predict if all words in context have saliency scores
                    winner_word = context.iloc[scores.index(max(scores))]
                    pred_end_relative_position = winner_word['distance']
                    pred_end_letter_relative_position = find_letter_distance_2centre_of_context_word(winner_word,
                                                                                                     letter_map)
                    # print(f'context: {" ".join([row.context_ia for row in context.itertuples()])}')
                    # print(f'fixated word: {context["ia"].tolist()[0]}')
                    # print(f'norm_length: {context["norm_length"].tolist()}')
                    # print(f'norm_entropy: {context["norm_entropy"].tolist()}')
                    # print(f'norm_surprisal: {context["norm_surprisal"].tolist()}')
                    # print(f'similarity: {context["similarity"].tolist()}')
                    # print(f'saliency scores: {scores}')
                    # print(f'winner word: {winner_word["context_ia"]}')
                    # print(f'predicted end relative word position: {pred_end_relative_position}')
                    # print(f'predicted end relative letter position: {pred_end_letter_relative_position}')
                    # exit()

            elif variable == 'combi_dist_len_sur_en_ss':
                pred_end_relative_position, pred_end_letter_relative_position = None, None
                scores = compute_combi_len_ss(context, ep=.5, weight_type='distance')
                if not np.isnan(scores).any():  # only predict if all words in context have saliency scores
                    winner_word = context.iloc[scores.index(max(scores))]
                    pred_end_relative_position = winner_word['distance']
                    pred_end_letter_relative_position = find_letter_distance_2centre_of_context_word(winner_word,
                                                                                                     letter_map)
                    # print(f'context: {" ".join([row.context_ia for row in context.itertuples()])}')
                    # print(f'fixated word: {context["ia"].tolist()[0]}')
                    # print(f'norm_length: {context["norm_length"].tolist()}')
                    # print(f'norm_entropy: {context["norm_entropy"].tolist()}')
                    # print(f'norm_surprisal: {context["norm_surprisal"].tolist()}')
                    # print(f'similarity: {context["similarity"].tolist()}')
                    # print(f'saliency scores: {scores}')
                    # print(f'winner word: {winner_word["context_ia"]}')
                    # print(f'predicted end relative word position: {pred_end_relative_position}')
                    # print(f'predicted end relative letter position: {pred_end_letter_relative_position}')
                    # exit()

            elif variable == 'combi_mass_len_sur_en_ss':
                pred_end_relative_position, pred_end_letter_relative_position = None, None
                scores = compute_combi_len_ss(context)
                word_distances = [position - 1 for position in context['distance'].tolist()]
                # find letter distances to fixation
                letter_distances = []
                for context_word in context.itertuples():
                    letter_distance = find_letter_distance_2centre_of_context_word(context_word, letter_map)
                    letter_distances.append(letter_distance)
                if not np.isnan(scores).any():
                    pred_end_relative_position = centre_of_mass(scores, word_distances)
                    pred_end_letter_relative_position = centre_of_mass(scores, letter_distances)

            saliency_variables['pred_end_relative_position'].append(pred_end_relative_position)
            saliency_variables['pred_end_letter_relative_position'].append(pred_end_letter_relative_position)

    saliency_df = pd.DataFrame.from_dict(saliency_variables)
    saliency_df.to_csv(filepath, index=False)

def main():

    model_name = 'gpt2'
    layers = '11'
    corpus_name = 'meco'
    eye_data_filepath = f'data/processed/{corpus_name}/{model_name}/full_{model_name}_[{layers}]_{corpus_name}_window_df.csv'
    words_filepath = f'data/processed/{corpus_name}/words_en_df.csv'
    data_split = ['validation'] # test

    eye_data = pd.read_csv(eye_data_filepath, index_col=0)
    words_data = pd.read_csv(words_filepath, index_col=0)

    val_eye_data, val_word_data, test_eye_data, test_word_data = split_data(eye_data, words_data)
    datasets = {'validation': [val_eye_data, val_word_data], 'test': [test_eye_data, test_word_data]}

    for split in data_split:
        if split in datasets.keys():
            output_filepath = f'data/processed/{corpus_name}/{model_name}/saliency_{model_name}_[{layers}]_{corpus_name}_{split}.csv'
            compute_saliency(datasets[split][0], datasets[split][1], output_filepath)
            # saliency_df = pd.read_csv(output_filepath, index_col=0)

if __name__ == '__main__':
    main()