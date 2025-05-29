import json

import torch
from torch.utils.data import Dataset
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict
from saliency_utils import normalize, baseline_7letter_2right, compute_letter_map, find_letter_distances, find_letter_distance_2centre_of_context_word

def remove_fixations_nan(data):

    fixations_to_exclude = []

    for i, context in data.groupby(['participant_id', 'trialid', 'fixid']):
        for context_word in context.itertuples():
            if (math.isnan(context_word.similarity)
                    or math.isnan(context_word.surprisal)
                    or math.isnan(context_word.entropy)
                    or math.isnan(context_word.length)):
                fixations_to_exclude.append((i[0],i[1],i[2]))

    data['fixid'] = data.apply(lambda x: np.nan if (x['participant_id'], x['trialid'], x['fixid']) in fixations_to_exclude else x['fixid'], axis=1).copy()
    data.dropna(subset=['fixid'], inplace=True)

    return data

def pre_process_fixation_data(fixation_data, norm_method='max-min'):

    # remove fixations with nan values in feature columns
    print('Remove fixations with nan values...')
    fixation_data = remove_fixations_nan(fixation_data)

    # normalize variables for combination computation
    print('Normalizing features...')
    for feature in ['length', 'entropy', 'surprisal']:
        norm_feature = normalize(fixation_data[feature].tolist(), norm_method)
        fixation_data[f'norm_{feature}'] = norm_feature
    # convert entropy values from previous context to 0.0001
    fixation_data['norm_entropy'] = fixation_data.apply(
        lambda x: 1e-06 if x['distance'] in [-3, -2, -1, 0] else x['norm_entropy'], axis=1)

    return fixation_data

def compute_input_arrays(fixation_data, filepath, letter_map=None, features='similarity,length,entropy,surprisal'):

    all_features = []

    for i, context in fixation_data.groupby(['participant_id', 'trialid', 'fixid']):

        fixation_features = defaultdict(list)
        pos_letter7_2right = None

        if letter_map and '7letter_2right' in features:
            pos_letter7_2right = baseline_7letter_2right(context, letter_map, level_type='word') # find which word and letter 7 letters to the right

        for context_word in context.itertuples():

            if '7letter_2right' in features and pos_letter7_2right:
                # if word is 7 letter to the right
                if context_word.distance == pos_letter7_2right:
                    fixation_features['7letter_2right'].append(1.)
                else:
                    fixation_features['7letter_2right'].append(0.)

            if 'similarity' in features:
                fixation_features['similarity'].append(context_word.similarity)
            if 'length' in features:
                fixation_features['length'].append(context_word.norm_length)
            if 'entropy' in features:
                fixation_features['entropy'].append(context_word.norm_entropy)
            if 'surprisal' in features:
                fixation_features['surprisal'].append(context_word.norm_surprisal)

        all_features.append([feature_list for feature_list in fixation_features.values()])

    x = torch.tensor(all_features)
    torch.save(x, filepath)

def compute_true_target_arrays(fixation_data, filepath, level_type='word'):

    end_positions = []

    for i, context in fixation_data.groupby(['participant_id', 'trialid', 'fixid']):

        for context_word in context.itertuples():
            # register true landing position
            if context_word.landing_target:
                if level_type == 'letter':
                    end_positions.append(context_word.letter_distance)
                else:  # level_type = word
                    end_positions.append(context_word.distance)

    y = torch.tensor(end_positions)
    torch.save(y, filepath)

def compute_letter_position_arrays(fixation_data, filepath, letter_map):

    letter_positions = []

    for i, context in fixation_data.groupby(['participant_id', 'trialid', 'fixid']):
        # register letter distances
        context_letter_positions = find_letter_distances(context, letter_map, shift_centre=True)
        letter_positions.append(context_letter_positions)

    letter_positions = torch.tensor(letter_positions)
    torch.save(letter_positions, filepath)

def compute_baseline_arrays(fixation_data, filepath, level_type, letter_map, base_type='next_word'):

    pred_end_positions = []

    for i, context in fixation_data.groupby(['participant_id', 'trialid', 'fixid']):
        if base_type == '7letter_2right':
            pred_end_relative_position = baseline_7letter_2right(context, letter_map, level_type)
        else:  # base_type = next_word
            pred_end_relative_position = 1
            if level_type == 'letter':
                letter_distance = None
                for context_word in context.itertuples():
                    if context_word.distance == 1:
                        letter_distance = find_letter_distance_2centre_of_context_word(context_word, letter_map)
                pred_end_relative_position = letter_distance
        pred_end_positions.append(pred_end_relative_position)

    y_base = torch.tensor(pred_end_positions)
    torch.save(y_base, filepath)

def convert_data_to_tensors(eye_data, word_data, opt_dir, level='word', features='similarity,length,entropy,surprisal', pre_process=False, data_filepath=''):

    if pre_process:
        print('Pre-processing data...')
        eye_data = pre_process_fixation_data(eye_data)
        if data_filepath:
            eye_data.to_csv(data_filepath.replace('_df.csv', '_cleaned.csv'), index=False)

    letter_map = compute_letter_map(word_data)
    with open(f'{opt_dir}/letter_map.json', 'w') as f:
        json.dump(letter_map, f, indent=4)

    # compute feature and true target arrays
    for text_id in eye_data['trialid'].unique():

        text_eye_data = eye_data[eye_data['trialid'] == text_id].copy()

        x_filepath = f'{opt_dir}/x_{text_id}_tensor.pt'
        y_filepath = f'{opt_dir}/y_{text_id}_{level}_tensor.pt'

        if not os.path.exists(x_filepath):
            print(f'Computing x arrays of text {text_id}...')
            compute_input_arrays(text_eye_data, x_filepath, letter_map, features)
        if not os.path.exists(y_filepath):
            print(f'Computing y arrays of text {text_id}...')
            compute_true_target_arrays(text_eye_data, y_filepath, level)
        if level == 'letter':
            letter_filepath = f'{opt_dir}/letter_positions_{text_id}_tensor.pt'
            if not os.path.exists(letter_filepath):
                print(f'Computing letter position arrays of text {text_id}...')
                compute_letter_position_arrays(text_eye_data, letter_filepath, letter_map)
        for baseline in ['next_word']: # '7letter_2right'
            y_base_filepath = f'{opt_dir}/y_{text_id}_{baseline}_{level}_tensor.pt'
            if not os.path.exists(y_base_filepath):
                print(f'Computing {baseline} baseline arrays of text {text_id}...')
                compute_baseline_arrays(text_eye_data, y_base_filepath, level, letter_map, baseline)

def split_data(text_ids, split_type='cross-validation', n_splits=5, test_size=.2, shuffle=False, random_state=42, filepath=''):

    splits = []

    if split_type == 'cross-validation':
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_index, test_index in kf.split(text_ids):
            splits.append({'train_index': train_index, 'test_index': test_index})

    else:
        train, test = train_test_split(text_ids, test_size=test_size, shuffle=shuffle, random_state=random_state)
        splits.append({'train_index': train, 'test_index': test})

    if filepath:
        with open(filepath, 'w') as f:
            f.write('split\ttrain\ttest\n')
            for i, split in enumerate(splits):
                f.write(f'{i}\t{split["train_index"]}\t{split["test_index"]}\n')

    return splits

def load_tensors(filepaths):

    all_arrays = []

    if filepaths:
        for filepath in filepaths:
            tensor = torch.load(filepath)
            all_arrays.append(tensor)

        # shape = (n_fixations, n_features, context_window_size) if x tensors
        # shape = (n_fixations) if y tensors
        tensor = torch.cat(all_arrays, dim=0)
    else:
        tensor = None

    return tensor

def load_baseline_tensors(split, baseline, opt_dir, level='word'):

    baseline_filepaths = []
    for text_id in split:
        baseline_filepaths.append(f'{opt_dir}/y_{text_id}_{baseline}_{level}_tensor.pt')

    predicted = load_tensors(baseline_filepaths)

    return predicted

def clean_tensors(true_targets, pred_targets):

    # remove possible NaN values
    nan_indices = torch.isnan(pred_targets)
    if nan_indices.any():
        pred_targets = pred_targets[~nan_indices]
        true_targets = true_targets[~nan_indices]

    # remove possible inf values
    inf_indices = torch.isinf(pred_targets)
    if inf_indices.any():
        pred_targets = pred_targets[~inf_indices]
        true_targets = true_targets[~inf_indices]

    return true_targets, pred_targets

def compute_split_arrays(trialids, directory, level='word',
                         features='similarity,length,entropy,surprisal',
                         concatenate=True, class_indices=True,
                         random=False):

    x_filepaths, y_filepaths, letter_pos_filepaths = [], [], []

    for i, text_id in enumerate(trialids):

        x_filepaths.append(f'{directory}/x_{text_id}_tensor.pt')
        y_filepaths.append(f'{directory}/y_{text_id}_{level}_tensor.pt')
        if level == 'letter':
            letter_pos_filepaths.append(f'{directory}/letter_positions_{text_id}_tensor.pt')

    x_tensor = load_tensors(x_filepaths)
    y_tensor = load_tensors(y_filepaths)
    letpos_tensor = load_tensors(letter_pos_filepaths)

    # remove feature(s) not selected
    x_data = []
    for fixation_array in x_tensor:
        fixation_features = []
        # MAKE SURE THE ORDER OF THE FEATURES IS THE SAME AS THE ORDER OF FEATURES IN THE INPUT VECTORS
        for feature_array, feature in zip(fixation_array, ['similarity', 'length', 'entropy', 'surprisal']): # 7letter_2right
            if feature in features.split(','):
                fixation_features.append(feature_array)
        fixation_features = torch.stack(fixation_features)
        x_data.append(fixation_features)
    x_tensor = torch.stack(x_data)

    if concatenate:
        # concatenate features so that x shape is (n_fixations, n_features*context_window_size)
        x_tensor = x_tensor.flatten(1,2)

    if class_indices:
        # change labels to class indices (as required by pytorch)
        y_tensor = y_tensor + torch.tensor([3])

    if random:
        x_tensor = torch.randn(x_tensor.shape)

    assert x_tensor.shape[0] == y_tensor.shape[0]

    return x_tensor, y_tensor, letpos_tensor

def compute_participant_indices(eye_data, split_ids):

    participant_indices = defaultdict(list)
    counter = 0
    for text_id in split_ids:
        text_data = eye_data[eye_data['trialid']==text_id].copy()
        for participant, fixation in text_data.groupby(['participant_id', 'fixid']):
            participant_indices[participant[0]].append(counter)
            counter += 1

    return participant_indices

def compute_participant_split_array(participant_indices, x_tensor, y_tensor, letpos_tensor=None):

    participant_indices = torch.tensor(participant_indices)
    x_tensor = torch.index_select(x_tensor, dim=0, index=participant_indices)
    y_tensor = torch.index_select(y_tensor, dim=0, index=participant_indices)
    if letpos_tensor:
        letpos_tensor = torch.index_select(letpos_tensor, dim=0, index=participant_indices)

    return x_tensor, y_tensor, letpos_tensor


class FixationDataset(Dataset):

  def __init__(self, text_IDs, opt_dir, features='similarity,length,surprisal,entropy', random=False):

        self.x_tensor, self.y_tensor, _ = compute_split_arrays(trialids=text_IDs, directory=opt_dir, features=features,
                                                               concatenate=True, class_indices=True,
                                                               random=random)
        self.item_IDs = range(self.x_tensor.shape[0])

  def __len__(self):

        return len(self.item_IDs)

  def __getitem__(self, index):

        # Select sample
        ID = self.item_IDs[index]

        # Load data and get label
        x = self.x_tensor[ID]
        y = self.y_tensor[ID]

        return x, y
