import pandas as pd
import numpy as np
from collections import defaultdict
import math
import json
from contextual_semantic_similarity.process_corpus import check_alignment

def fix_misalignment_letter_map(row, letter_ids):

    if row.trialid == 0 and row.ianum > 20:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 40:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 62:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 79:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 96:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 114:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 132:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 149:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 163:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 0 and row.ianum > 180:
        letter_ids = [p - 1 for p in letter_ids]

    if row.trialid == 1 and row.ianum > 5:
        letter_ids = [p + 1 for p in letter_ids]
    if row.trialid == 1 and row.ianum == 7:
        letter_ids = [p if p < 47 else 48 for p in letter_ids]

    if row.trialid == 2 and row.ianum > 9:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 2 and row.ianum > 16:
        letter_ids = [p - 1 for p in letter_ids]

    if row.trialid == 3 and row.ianum > 17:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 3 and row.ianum > 33:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 3 and row.ianum > 36:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 3 and row.ianum > 53:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 3 and row.ianum > 75:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 3 and row.ianum > 91:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 3 and row.ianum > 108:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 3 and row.ianum > 124:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 3 and row.ianum > 141:
        letter_ids = [p - 1 for p in letter_ids]

    if row.trialid == 4 and row.ianum > 18:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 4 and row.ianum > 36:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 4 and row.ianum > 51:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 4 and row.ianum > 67:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 4 and row.ianum > 83:
        letter_ids = [p - 1 for p in letter_ids]
    if row.trialid == 4 and row.ianum > 85:
        letter_ids = [p - 1 for p in letter_ids]
    # if row.trialid == 4 and row.ianum > 94:
    #     letter_ids = [p - 1 for p in letter_ids]


    return letter_ids

def compute_letter_map(words_df):

    # map each word in each text to its letter indices at the text level
    letter_map = defaultdict(list)

    for i, text in words_df.groupby('trialid'):
        position_in_text = 0
        for row in text.itertuples():
            all_letter_ids = [i + 1 for i, character in enumerate(row.texts)]
            letter_ids = all_letter_ids[position_in_text:position_in_text + row.length]
            # letter_ids = fix_misalignment_letter_map(row, letter_ids)
            position_in_text += row.length + 1
            letter_map[f'{int(row.trialid)}-{int(row.ianum)}'] = letter_ids

    return letter_map

def check_alignment_letter_map(letter_map, fixation_df):

    for i, rows in fixation_df.groupby('trialid'):
        rows = rows.sort_values(by='ianum')
        for row in rows.itertuples():
                match = False
                fixated_letter = row.letter
                if (fixated_letter not in ["", " ", '"'] and
                        (row.trialid != 1 and row.ianum != 7) and
                        (row.trialid != 2 and row.ianum != 33) and
                        (row.trialid != 3 and row.ianum != 157)):
                    fixated_word = row.ia
                    if row.ianum > 0:
                        fixated_word = ' ' + fixated_word
                    if row.ianum < max(rows['ianum'].tolist()):
                        fixated_word = fixated_word + ' '
                    fixated_letter_occs = [p + 1 for p, l in enumerate(fixated_word) if l == fixated_letter]
                    ia_letter_ids = []
                    for index in fixated_letter_occs:
                        spaces = [l for l in fixated_word if l == ' ']
                        ia_letter_ids = [row.letternum + (lp - index) for lp in range(len(spaces), len(fixated_word))]
                        if letter_map[f"{row.trialid}-{row.ianum}"] == ia_letter_ids:
                            match = True
                            break
                    if not match:
                        raise Exception(f'Letter map {letter_map[f"{row.trialid}-{row.ianum}"]} '
                                        f'for {row.ia} (id {row.ianum}) in trial {row.trialid} from participant '
                                        f'{row.participant_id} does not match '
                                        f'letter number {row.letternum} for the same word in fixation report {ia_letter_ids}.')

def align_letter_indices(eye_df, letter_map):

    eye_df.sort_values(by=['trialid', 'ianum'], inplace=True)
    letternum = []
    for row in eye_df.itertuples():
        letter_indices = letter_map[f'{row.trialid}-{row.ianum}']
        if row.letter not in [" ", '"']:
            fixated_letter_idx_at_word = row.ia.index(row.letter) # first occurrence (simplification)
            fixated_letter_idx = letter_indices[fixated_letter_idx_at_word]
            letternum.append(fixated_letter_idx)
        else:
            letternum.append(float('nan'))
    eye_df['letternum'] = letternum
    eye_df.sort_values(by=['participant_id', 'trialid', 'fixid'], inplace=True)

def find_letter_distance_to_fixation(context, letter_map):

    centre_let_pos_context_word = None
    for context_word in context.itertuples():
        if context_word.distance == 1:
            all_let_pos_context_word = letter_map[f'{context_word.trialid}-{context_word.context_ianum}']
            if all_let_pos_context_word:
                centre_let_pos_context_word = all_let_pos_context_word[
                    math.ceil(len(all_let_pos_context_word) / 2) - 1]
    return centre_let_pos_context_word

def find_letter_distance_2centre_of_context_word(context_word, letter_map, shifted_centre=None):

    # compute distance in letters from fixated letter to the centre of context word
    all_let_pos_context_word = letter_map[f'{context_word.trialid}-{context_word.context_ianum}']
    if all_let_pos_context_word:
        centre_let_pos_context_word = all_let_pos_context_word[math.ceil(len(all_let_pos_context_word) / 2) - 1]
        if shifted_centre: # reference point is the centre of next word (to skew attention to n+1)
            let_pos_fix_word = shifted_centre
        else: # reference point is letter being fixated
            let_pos_fix_word = context_word.letternum
        letter_distance = centre_let_pos_context_word - let_pos_fix_word
    else:
        raise ValueError(f'{context_word.trialid},{context_word.context_ianum} is empty in letter map \n {context_word}')

    return letter_distance

def find_letter_distances(context, letter_map, shift_centre=True):

    letter_distances = []
    centre_let_pos_context_word = None

    if shift_centre:
        # because we consider the centre of n+1 as reference point (instead of letter being fixated at n)
        # this is done to ensure the centre of fixations becomes n+1 instead of n
        centre_let_pos_context_word = find_letter_distance_to_fixation(context, letter_map)

    for context_word in context.itertuples():
        letter_distance = find_letter_distance_2centre_of_context_word(context_word, letter_map,
                                                                       shifted_centre=centre_let_pos_context_word)
        letter_distances.append(letter_distance)

    return letter_distances

def baseline_7letter_2right(context, letter_map, level_type):

    # word predicted is 7 letters to the right of the fixated letter,
    # and logically the letter predicted is 7 letters to the right of fixated letter
    # which word is 7 letters to the right of fixated letter
    pos_letter7_2right = float('nan') # None
    if level_type == 'letter':
        pos_letter7_2right = 7.0
    else:
        # optimal saccade distance visual field
        distance_letter7_2right = 7.0
        # position of letter being fixated
        fix_letter = context['letternum'].tolist()[0]
        # find position of letter 7 letter positions to the right of the letter being fixated
        letter7_2right = fix_letter + distance_letter7_2right
        for context_word in context.itertuples():
            # find letter positions of context word
            let_positions = letter_map[f'{context_word.trialid}-{context_word.context_ianum}']
            # check if the letter position 7 letters to the right is in this context word
            if letter7_2right in let_positions:
                pos_letter7_2right = context_word.distance
                assert pos_letter7_2right > -1, print('Word 7 letters to the right is to the left of the fixated word. This is not possible.')

    return pos_letter7_2right

def winner_takes_all(context, letter_map, level_type, feature, condition, weight_type='raw'):

    winner_score, winner, winner_pos = None, None, None
    distance_weights_max = {-3:.25, -2:.50, -1:.75, 0:.75, 1:1, 2:.75, 3:.5}
    distance_weights_min = {-3:1, -2:.75, -1:.50, 0:.50, 1:.25, 2:.50, 3:.75}

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
        if level_type == 'letter':
            winner_pos = find_letter_distance_2centre_of_context_word(winner, letter_map)
        else:
            winner_pos = winner.distance

    # print('Winner: ', winner)
    # print('Winner position: ', winner_pos, winner_let_pos)

    return winner_pos

def centre_of_mass(saliencies, positions):

    return (1/np.sum(saliencies))*np.sum([saliency*position for saliency,position in zip(saliencies,positions)])

def normalize(all_values, method='max-min'):

    if method == 'z-score':
        norm_feature = (all_values - np.mean(all_values)) / np.std(all_values)

    else:
        norm_feature = []
        min_feature = min([x for x in all_values if not math.isnan(x)])
        max_feature = max([x for x in all_values if not math.isnan(x)])

        for x in all_values:
            norm = None
            if x:
                norm = (x - min_feature) / (max_feature - min_feature)
                # if norm == 0.0:
                #     norm = 1e-06 # to avoid 0 for minimum
            norm_feature.append(norm)

    return norm_feature

def compute_combi_len_ss(context, pos_letter7_2right, x=None, mapping_type='raw'):

    scores = []
    letter7_2right = 0.
    distance_weights = {-3: .25, -2: .50, -1: .75, 0: .75, 1: 1, 2: .75, 3: .5}

    if not x:
        x = [1., 1., 1., 1., 1.]

    for context_word in context.itertuples():

        features = {'similarity': context_word.similarity,
                    'entropy': context_word.norm_entropy,
                    'length': context_word.norm_length,
                    'surprisal': context_word.norm_surprisal}

        if context_word.distance == pos_letter7_2right:
            letter7_2right = 1.

        # compute combi
        if not np.isnan(list(features.values())).any():
            score = ((-context_word.similarity * x[0])
                         + (context_word.norm_length * x[1])
                         + (context_word.norm_surprisal * x[2])
                         + (context_word.norm_entropy * x[3])
                         + (letter7_2right * x[4]))
            # weight by distance
            if mapping_type == 'distance':
                score = score * distance_weights[context_word.distance]
        else:
            score = np.nan
        scores.append(score)

    return scores

def compute_saliency(df, filepath, saliency_types, letter_map, level_type='word', weights=None, normalize_predictions=False):

    saliency_variables = {'participant_id': [],
                          'text_id': [],
                          'context_window': [],
                          'start_ia': [],
                          'start_position' : [],
                          'end_ia': [],
                          'end_position' : [],
                          'saliency_type': [],
                          'end_relative_position': [],
                          'pred_end_relative_position': []}

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
                    if level_type == 'letter':
                        saliency_variables['end_relative_position'].append(context_word.letter_distance)
                    else:
                        saliency_variables['end_relative_position'].append(context_word.distance)

            # for computation of saliency formula
            pos_letter7_2right = baseline_7letter_2right(context, letter_map, level_type='word')

            # compute end relative position using saliency
            pred_end_relative_position = None

            # baseline n+1
            if variable == 'next_word':
                pred_end_relative_position = 1
                if level_type == 'letter':
                    letter_distance = None
                    for context_word in context.itertuples():
                        if context_word.distance == 1:
                            letter_distance = find_letter_distance_2centre_of_context_word(context_word, letter_map)
                    pred_end_relative_position = letter_distance

            # baseline 7 letters to right of center of fixation
            elif variable == '7letter_2right':
                pred_end_relative_position = baseline_7letter_2right(context, letter_map, level_type)

            # longest word wins
            elif variable == 'max_length':
                pred_end_relative_position = winner_takes_all(context, letter_map, level_type,
                                                              feature='length', condition="max")

            # longest word weighted by distance
            elif variable == 'dist_max_length':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context, letter_map, level_type,
                                                                                                  feature='length',
                                                                                                  condition="max",
                                                                                                  weight_type='distance')
            # less frequent word wins
            elif variable == 'min_frequency':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context,letter_map, level_type,
                                                                                                feature ='frequency',
                                                                                                condition="min")
            # less frequent weighted by distance
            elif variable == 'dist_min_frequency':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context, letter_map, level_type,
                                                                            feature='frequency',
                                                                            condition="min",
                                                                            weight_type='distance')
            # more surprising word wins
            elif variable == 'max_surprisal':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context,letter_map, level_type,
                                                                                                feature ='surprisal',
                                                                                                condition="max")
            # more surprising weighted by distance
            elif variable == 'dist_max_surprisal':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context, letter_map, level_type,
                                                                                                feature='surprisal',
                                                                                                condition="max",
                                                                                                weight_type='distance')
            # less similar word wins
            elif variable == 'min_semantic_similarity':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context,letter_map, level_type,
                                                                                                 feature ='similarity',
                                                                                                 condition="min")
            # less similar weighted by distance
            elif variable == 'dist_min_semantic_similarity':
                pred_end_relative_position, pred_end_letter_relative_position = winner_takes_all(context,letter_map, level_type,
                                                                                                 feature ='similarity',
                                                                                                 condition="min",
                                                                                                 weight_type='distance')

            # centre of mass
            # TODO adjust positions and add letter positions
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
                pred_end_relative_position= None
                scores = compute_combi_len_ss(context, pos_letter7_2right, weights)
                if not np.isnan(scores).any(): # only predict if all words in context have saliency scores
                    winner_word = context.iloc[scores.index(max(scores))]
                    pred_end_relative_position = winner_word['distance']
                    if level_type == 'letter':
                        pred_end_relative_position = find_letter_distance_2centre_of_context_word(winner_word,
                                                                                                         letter_map)

            elif variable == 'combi_dist_len_sur_en_ss':
                pred_end_relative_position = None
                scores = compute_combi_len_ss(context, pos_letter7_2right, weights, mapping_type='distance')
                if not np.isnan(scores).any():  # only predict if all words in context have saliency scores
                    winner_word = context.iloc[scores.index(max(scores))]
                    pred_end_relative_position = winner_word['distance']
                    if level_type == 'letter':
                        pred_end_relative_position = find_letter_distance_2centre_of_context_word(winner_word,
                                                                                                     letter_map)

            elif variable == 'combi_mass_len_sur_en_ss':
                pred_end_relative_position = None
                scores = compute_combi_len_ss(context, pos_letter7_2right, weights)
                word_distances = [position - 1 for position in context['distance'].tolist()]
                # find letter distances to fixation
                letter_distances = find_letter_distances(context, letter_map)
                if not np.isnan(scores).any():
                    if level_type == 'letter':
                        pred_end_relative_position = centre_of_mass(scores, letter_distances)
                    else:
                        pred_end_relative_position = centre_of_mass(scores, word_distances)

            saliency_variables['pred_end_relative_position'].append(pred_end_relative_position)

    if normalize_predictions:
        max_pos = max([pos for pos in saliency_variables['end_relative_position'] if not math.isnan(pos)])
        saliency_variables['end_relative_position'] = [pos / max_pos for pos in saliency_variables['end_relative_position']]
        saliency_variables['pred_end_relative_position'] = [pos / max_pos for pos in saliency_variables['pred_end_relative_position']]

    saliency_df = pd.DataFrame.from_dict(saliency_variables)
    saliency_df.to_csv(filepath, index=False)
#
# def main():
#
#     print('Preparing to compute saliency...')
#
#     model_name = 'gpt2'
#     layers = '11'
#     corpus_name = 'meco'
#     eye_data_filepath = f'data/processed/{corpus_name}/{model_name}/full_{model_name}_[{layers}]_{corpus_name}_window_df.csv'
#     words_filepath = f'data/processed/{corpus_name}/words_en_df.csv'
#     saliency_types = ['next_word', '7letter_2right', 'combi_mass_len_sur_en_ss']
#     level_type = 'letter'
#     weights = []
#
#     eye_data = pd.read_csv(eye_data_filepath, index_col=0)
#     words_data = pd.read_csv(words_filepath, index_col=0)
#
#     print('Computing letter distances...')
#     letter_map = compute_letter_map(words_data)
#
#     print('Normalizing features...')
#     for feature in ['length', 'entropy', 'surprisal']:
#         norm_feature = normalize(eye_data[feature].tolist())
#         eye_data[f'norm_{feature}'] = norm_feature
#     # convert entropy values from previous context to 0.0001
#     eye_data['norm_entropy'] = eye_data.apply(lambda x: 0.0001 if x['distance'] in [-3, -2, -1, 0] else x['norm_entropy'], axis=1)
#
#     print('Compute saliency values...')
#     output_filepath = f'data/processed/{corpus_name}/{model_name}/saliency_{saliency_types}_{level_type}_{model_name}_[{layers}]_{corpus_name}.csv'
#     compute_saliency(eye_data, output_filepath, saliency_types, letter_map, level_type, weights, normalize_predictions=True)
#
# if __name__ == '__main__':
#     main()

# eye_data_filepath = f'data/processed/meco/gpt2/full_gpt2_[11]_meco_window_cleaned.csv'
# eye_data_filepath = f'data/processed/meco/fixation_report_en_df.csv'
# words_filepath = f'data/processed/meco/words_en_df.csv'
# eye_data = pd.read_csv(eye_data_filepath)
# words_data = pd.read_csv(words_filepath)
# letter_map = compute_letter_map(words_data)
# with open('letter_map.json', 'w') as fp:
#     json.dump(letter_map, fp, indent=4)
# check_alignment_letter_map(letter_map,eye_data)
# align_letter_indices(eye_data, letter_map)