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
import pickle

def remove_fixations_nan(data):

    fixations_to_exclude = []

    for i, context in data.groupby(['participant_id', 'trialid', 'fixid']):
        for context_word in context.itertuples():
            if (math.isnan(context_word.similarity)
                    or math.isnan(context_word.surprisal)
                    or math.isnan(context_word.entropy)
                    or math.isnan(context_word.length)
                    or math.isnan(context_word.previous_sacc_distance)
                    or math.isnan(context_word.previous_fix_duration)
                    or context_word.pos_tag == ''
                    or math.isnan(context_word.frequency)
                    or math.isnan(context_word.has_been_fixated)):
                fixations_to_exclude.append((i[0],i[1],i[2]))

    data['fixid'] = data.apply(lambda x: np.nan if (x['participant_id'], x['trialid'], x['fixid']) in fixations_to_exclude else x['fixid'], axis=1).copy()
    data.dropna(subset=['fixid'], inplace=True)

    return data

def pre_process_fixation_data(fixation_data, norm_method='max-min'):

    # remove fixations with nan values in feature columns
    print('Remove fixations with nan values...')
    fixation_data = remove_fixations_nan(fixation_data)
    # category features: from str to int
    pos_tag_map = defaultdict(int)
    counter = 0
    for pos_tag in fixation_data['pos_tag'].unique():
        pos_tag_map[pos_tag] = counter
        counter += 1
    fixation_data['pos_tag_index'] = fixation_data['pos_tag'].map(pos_tag_map)
    # normalize variables for combination computation
    print('Normalizing features...')
    for feature in ['length', 'similarity', 'entropy', 'surprisal', 'distance', 'previous_sacc_distance',
                    'previous_fix_duration', 'pos_tag_index', 'frequency']:
        norm_feature = normalize(fixation_data[feature].tolist(), norm_method)
        fixation_data[f'norm_{feature}'] = norm_feature
    # convert entropy values from previous context to minimum
    # min_entropy = 1e-06
    min_entropy = fixation_data['norm_entropy'].min()
    fixation_data['norm_entropy'] = fixation_data.apply(
        lambda x: min_entropy if x['distance'] in [-3, -2, -1, 0] else x['norm_entropy'], axis=1)

    return fixation_data

def compute_token_embedding(text_embeddings, context_word, word_to_token_map):

    if word_to_token_map:
        # embedding of sequence up to fixated word
        sequence_embedding = text_embeddings[context_word.ianum]
        # find positions of tokens which correspond to the context word
        token_positions = word_to_token_map[context_word.trialid][context_word.context_ianum]
        # take embeddings of tokens
        token_embeddings = sequence_embedding[token_positions[0]:token_positions[-1] + 1]
        # if word made of more than one token, average them
        word_embedding = torch.mean(token_embeddings, dim=0)
    else:
        raise ValueError('word_to_token_map must be provided if embedding in features.')

    return word_embedding

def compute_input_arrays(fixation_data, filepath, letter_map=None,
                         features='similarity,length,entropy,surprisal', word_to_token_map=None, vectors_dir=''):

    all_word_features = []
    all_fix_features = []
    all_embed_features = []

    for i, context in fixation_data.groupby(['participant_id', 'trialid', 'fixid']):

        if 'embedding' in features:
            text_embeddings = torch.load(f'{vectors_dir}/text{i[1]}_embeddings.pt')
            text_embeddings = text_embeddings.detach() # requires_grad = False

        word_features, fix_features = [],[]
        embed_features = torch.empty((4,768)) # 4 words and 768-dimension word embedding
        pos_letter7_2right = None

        if letter_map and '7letter_2right' in features:
            # find which word and letter 7 letters to the right
            pos_letter7_2right = baseline_7letter_2right(context, letter_map, level_type='word')

        for i_c, context_word in enumerate(context.itertuples()):

            context_word_text_features = []

            if '7letter_2right' in features and pos_letter7_2right:
                # if word is 7 letter to the right
                if context_word.distance == pos_letter7_2right:
                    context_word_text_features.append(1.)
                else:
                    context_word_text_features.append(0.)

            if 'similarity' in features:
                context_word_text_features.append(context_word.norm_similarity)
            if 'length' in features:
                context_word_text_features.append(context_word.norm_length)
            if 'entropy' in features:
                context_word_text_features.append(context_word.norm_entropy)
            if 'surprisal' in features:
                context_word_text_features.append(context_word.norm_surprisal)
            if 'pos_tag' in features:
                context_word_text_features.append(context_word.norm_pos_tag_index)
            if 'frequency' in features:
                context_word_text_features.append(context_word.norm_frequency)
            if 'has_been_fixated' in features:
                context_word_text_features.append(context_word.has_been_fixated)

            word_features.append(context_word_text_features)

            if 'embedding' in features and i_c < 4: # embeddings only for the previous words and fixated word
                word_embedding = compute_token_embedding(text_embeddings, context_word, word_to_token_map)
                embed_features[i_c] = word_embedding

        all_word_features.append(word_features)
        all_embed_features.append(embed_features)

        if 'previous_sacc_distance' in features:
            fix_features.append(context['norm_previous_sacc_distance'].tolist()[0])
        if 'previous_fix_duration' in features:
            fix_features.append(context['norm_previous_fix_duration'].tolist()[0])
        all_fix_features.append(fix_features)

        # fixation_features_unroll = []
        # for feature_list in fixation_features.values():
        #     fixation_features_unroll.extend(feature_list)
        # all_features.append(fixation_features_unroll)
        # all_features.append([feature_list for feature_list in fixation_features.values()])

    if all_word_features:
        x_text = torch.tensor(all_word_features)
        torch.save(x_text, f'{filepath}_word_features.pt')
    if all_fix_features:
        x_fix = torch.tensor(all_fix_features)
        torch.save(x_fix, f'{filepath}_fix_features.pt')
    if all_embed_features:
        x_embed = torch.stack(all_embed_features, dim=0)
        torch.save(x_embed, f'{filepath}_embed_features.pt')

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

def convert_data_to_tensors(eye_data, word_data, opt_dir, level='word', features='similarity,length,entropy,surprisal',
                            pre_process=False, norm_method='max-min', data_filepath='', word_to_token_map=None, vectors_dir=''):

    if pre_process:
        print('Pre-processing data...')
        eye_data = pre_process_fixation_data(eye_data, norm_method)
        if data_filepath:
            eye_data.to_csv(data_filepath.replace('_df.csv', '_cleaned.csv'), index=False)

    letter_map = compute_letter_map(word_data)
    with open(f'{opt_dir}/letter_map.json', 'w') as f:
        json.dump(letter_map, f, indent=4)

    # compute feature and true target arrays
    for _id, trial_data in eye_data.groupby(['participant_id', 'trialid']):

        x_filepath = f'{opt_dir}/x_{_id[0]}_{_id[1]}'
        y_filepath = f'{opt_dir}/y_{_id[0]}_{_id[1]}.pt' # {level}

        if not os.path.exists(x_filepath):
            print(f'Computing x arrays of participant {_id[0]}, text {_id[1]}...')
            compute_input_arrays(trial_data, x_filepath, letter_map, features, word_to_token_map, vectors_dir)
        if not os.path.exists(y_filepath):
            print(f'Computing y arrays of participant {_id[0]}, text {_id[1]}...')
            compute_true_target_arrays(trial_data, y_filepath, level)
        if level == 'letter':
            letter_filepath = f'{opt_dir}/letter_positions_{_id[0]}_{_id[1]}.pt'
            if not os.path.exists(letter_filepath):
                print(f'Computing letter position arrays of participant {_id[0]}, text {_id[1]}...')
                compute_letter_position_arrays(trial_data, letter_filepath, letter_map)
        for baseline in ['next_word']: # '7letter_2right'
            y_base_filepath = f'{opt_dir}/y_{_id[0]}_{_id[1]}_{baseline}_tensor.pt' # {level}
            if not os.path.exists(y_base_filepath):
                print(f'Computing {baseline} baseline arrays of participant {_id[0]}, text {_id[1]}...')
                compute_baseline_arrays(trial_data, y_base_filepath, level, letter_map, baseline)

def split_data(_ids, split_type='cross-validation', n_splits=5, test_size=.2, shuffle=False, random_state=42, filepath=''):

    splits = []

    if split_type == 'cross-validation':
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_index, test_index in kf.split(_ids):
            splits.append({'train_index': train_index, 'test_index': test_index})

    else:
        train, test = train_test_split(_ids, test_size=test_size, shuffle=shuffle, random_state=random_state)
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
            if os.path.exists(filepath):
                tensor = torch.load(filepath)
                all_arrays.append(tensor)
        # shape = (n_fixations, n_features, context_window_size) if x tensors
        # shape = (n_fixations) if y tensors
        tensor = torch.cat(all_arrays, dim=0)
    else:
        tensor = None

    return tensor

def load_baseline_tensors(split_ids, baseline, opt_dir, level='word'):

    baseline_filepaths = []

    for _id in split_ids:

        _id = _id.split(',')
        participant_id = _id[0]
        text_id = _id[1]

        baseline_filepaths.append(f'{opt_dir}/y_{participant_id}_{text_id}_{baseline}_tensor.pt') # {level}

    predicted = load_tensors(baseline_filepaths)

    return predicted

def prepare_feature_ablation(x_word_tensor, x_fix_tensor, x_embed_tensor, ablation_type, features_to_select, feature_map):

    # dropping features which are not selected
    if ablation_type == 'drop':

        word_features, fix_features = [], []
        x_word_splits, x_fix_splits = None, None

        if not torch.all(x_word_tensor == 0):
            x_word_splits = torch.split(x_word_tensor, 1, dim=-1)
        if not torch.all(x_fix_tensor == 0):
            x_fix_splits = torch.split(x_fix_tensor, 1, dim=-1)

        for feature in features_to_select.split(','):

            if feature in ['surprisal', 'frequency', 'length', 'has_been_fixated']:
                if feature_map:
                    if x_word_splits:
                        x_word_feature = x_word_splits[feature_map[feature]]
                        word_features.append(x_word_feature)
                    else:
                        raise ValueError(
                            'Word feature was selected for feature ablation, but word tensor is empty.')
                else:
                    raise ValueError(
                        'For feature ablation on word features, please provide also the feature map which determines the order of the features in the input vector.')

            elif feature in ['previous_fix_duration', 'previous_sacc_distance']:
                if feature_map:
                    if x_fix_splits:
                        x_fix_feature = x_fix_splits[feature_map[feature]]
                        fix_features.append(x_fix_feature)
                    else:
                        raise ValueError('Fixation feature was selected for feature ablation, but word tensor is empty.')
                else:
                    raise ValueError(
                        'For feature ablation on fixation features, please provide also the feature map which determines the order of the features in the input vector.')

        if word_features:
            x_word_tensor = torch.cat(word_features, dim=-1)
        else:
            x_word_tensor = torch.zeros(x_word_tensor.shape)

        if fix_features:
            x_fix_tensor = torch.cat(fix_features, dim=-1)
        else:
            x_fix_tensor = torch.zeros(x_fix_tensor.shape)

        if 'embedding' not in features_to_select:
            x_embed_tensor = torch.zeros(x_embed_tensor.shape)

    # mean-ablation: instead of dropping, replace features which are not selected with their averages
    elif ablation_type == 'mean':

        for feature in ['surprisal', 'frequency', 'length', 'has_been_fixated']:

            if feature not in features_to_select.split(','):

                if feature_map:
                    if not torch.all(x_word_tensor == 0):
                        if feature == 'has_been_fixated':
                            x_word_tensor[:, :, feature_map[feature]] = 0.0
                        else:
                            feature_mean = x_word_tensor[:,:,feature_map[feature]].mean()
                            x_word_tensor[:,:,feature_map[feature]] = feature_mean
                    else:
                        raise ValueError(
                            'Word feature was selected for feature ablation, but word tensor is empty.')
                else:
                    raise ValueError(
                        'For feature ablation on word features, please provide also the feature map which determines the order of the features in the input vector.')

        for feature in ['previous_fix_duration', 'previous_sacc_distance']:

            if feature not in features_to_select.split(','):

                if feature_map:
                    if not torch.all(x_fix_tensor == 0):
                        feature_mean = x_fix_tensor[:, feature_map[feature]].mean()
                        x_fix_tensor[:, feature_map[feature]] = feature_mean
                    else:
                        raise ValueError('Fixation feature was selected for feature ablation, but word tensor is empty.')
                else:
                    raise ValueError(
                        'For feature ablation on fixation features, please provide also the feature map which determines the order of the features in the input vector.')

        if 'embedding' not in features_to_select.split(','):
            feature_mean = x_embed_tensor.mean(dim=0)
            x_embed_tensor[:,] = feature_mean

    return x_word_tensor, x_fix_tensor, x_embed_tensor

def compute_split_arrays(_ids, directory, level='word', class_indices=True, random=False,
                         features_to_select='', feature_map=None, ablation_type='drop'):

    x_word_filepaths, x_fix_filepaths, x_embed_filepaths, y_filepaths, letter_pos_filepaths = [], [], [], [], []

    for i, _id in enumerate(_ids):

        _id = _id.split(',')
        participant_id = _id[0]
        text_id = _id[1]

        word_features = f'{directory}/x_{participant_id}_{text_id}_word_features.pt'
        fix_features = f'{directory}/x_{participant_id}_{text_id}_fix_features.pt'
        embed_features = f'{directory}/x_{participant_id}_{text_id}_embed_features.pt'

        if os.path.exists(word_features):
            x_word_filepaths.append(word_features)
        if os.path.exists(fix_features):
            x_fix_filepaths.append(fix_features)
        if os.path.exists(embed_features):
            x_embed_filepaths.append(embed_features)

        y_filepaths.append(f'{directory}/y_{participant_id}_{text_id}.pt') # {level}

        if level == 'letter':
            letter_pos_filepaths.append(f'{directory}/letter_positions_{participant_id}_{text_id}.pt')

    y_tensor = load_tensors(y_filepaths)

    x_word_tensor = load_tensors(x_word_filepaths)
    if x_word_tensor is None:
        x_word_tensor = torch.zeros(y_tensor.shape)

    x_fix_tensor = load_tensors(x_fix_filepaths)
    if x_fix_tensor is None:
        x_fix_tensor = torch.zeros(y_tensor.shape)

    x_embed_tensor = load_tensors(x_embed_filepaths)
    if x_embed_tensor is None:
        x_embed_tensor = torch.zeros(y_tensor.shape)

    letpos_tensor = load_tensors(letter_pos_filepaths)

    # Remove feature(s) not selected for feature ablation
    if features_to_select:
        x_word_tensor, x_fix_tensor, x_embed_tensor = prepare_feature_ablation(x_word_tensor, x_fix_tensor,
                                                                               x_embed_tensor, ablation_type,
                                                                               features_to_select, feature_map)

    if class_indices:
        # change labels to class indices (as required by pytorch)
        y_tensor = y_tensor + torch.tensor([3])

    if random:
        # compute random input vectors for random baseline
        x_word_tensor = torch.randn(x_word_tensor.shape)
        x_fix_tensor = torch.randn(x_fix_tensor.shape)
        x_embed_tensor = torch.randn(x_embed_tensor.shape)

    assert x_word_tensor.shape[0] == y_tensor.shape[0]
    assert x_fix_tensor.shape[0] == y_tensor.shape[0]
    assert x_embed_tensor.shape[0] == y_tensor.shape[0]

    return x_word_tensor, x_fix_tensor, x_embed_tensor, y_tensor, letpos_tensor

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

class FixationDataset(Dataset):

  def __init__(self,
               split_ids,
               _dir,
               features_to_select='',
               feature_map=None,
               random=False,
               ablation_type='drop'):

        self.x_word_tensor, self.x_fix_tensor, self.x_embed_tensor, self.y_tensor, _ = compute_split_arrays(_ids=split_ids, directory=_dir,
                                                                                                           features_to_select=features_to_select,
                                                                                                           feature_map=feature_map,
                                                                                                            class_indices=True, random=random,
                                                                                                            ablation_type=ablation_type)

        # remove requires_grad from the embeddings to allow parallel training (number of workers > 6)
        if self.x_embed_tensor.requires_grad:
            self.x_embed_tensor = self.x_embed_tensor.detach()
        self.x_embed_tensor = self.x_embed_tensor[:,3,:].squeeze() # contextual embedding of fixated word

        # flatten x_word_tensor
        self.x_word_tensor = torch.flatten(self.x_word_tensor, 1)

        if ablation_type == 'drop':
            inputs = []
            if not torch.all(self.x_word_tensor == 0):
                inputs.append(self.x_word_tensor)
            if not torch.all(self.x_fix_tensor == 0):
                inputs.append(self.x_fix_tensor)
            if not torch.all(self.x_embed_tensor == 0):
                inputs.append(self.x_embed_tensor)
            self.combined_input = torch.cat(inputs, -1)
        else:
            self.combined_input = torch.cat((self.x_word_tensor, self.x_fix_tensor, self.x_embed_tensor), -1)

        # split word features into previous and upcoming contexts
        # self.x_previous_context_tensor = self.x_word_tensor[:,:4,:]
        # self.x_upcoming_context_tensor = self.x_word_tensor[:,4:,:]
        # concatenate embeddings with previous context word features
        # self.x_previous_context_tensor = torch.cat((self.x_embed_tensor, self.x_previous_context_tensor), dim=-1)
        # # concatenate embeddings and low-level features for fcn
        # # (number of fixations, number of words, number of feature values
        # self.x_tensor = torch.cat((self.x_embed_tensor, self.x_word_tensor), dim=-1)
        # previous three words and currently fixated word have word all features
        # self.x_previous_context_tensor = self.x_tensor[:, :4, :]
        # # remove word dimension
        # previous_context_unrolled = torch.flatten(previous_context, start_dim=1)
        # upcoming context no word embedding
        # self.x_upcoming_context_tensor = self.x_tensor[:, 4:, 768:]
        # # remove word dimension
        # upcoming_context_unrolled = torch.flatten(upcoming_context, start_dim=1)
        # (number of fixations, number of feature values = 773*4 for the first 4 words + 5*3 for the last 3 words)
        # self.x_tensor = torch.cat((previous_context_unrolled, upcoming_context_unrolled), dim=-1)

        self.item_IDs = range(self.y_tensor.shape[0])

  def __len__(self):

        return len(self.item_IDs)

  def __getitem__(self, index):

        # Select sample
        ID = self.item_IDs[index]

        # # Load data
        # x_word = self.x_word_tensor[ID]
        # x_fix = self.x_fix_tensor[ID]
        # x_embed = self.x_embed_tensor[ID]
        # x = self.x_tensor[ID]
        # x_previous = self.x_previous_context_tensor[ID]
        # x_upcoming = self.x_upcoming_context_tensor[ID]
        # x_word = self.x_word_tensor[ID]
        # x_embed = self.x_embed_tensor[ID]
        # x_fix = self.x_fix_tensor[ID]

        x = self.combined_input[ID]

        # Load label
        y = self.y_tensor[ID]

        # return x_word, x_fix, x_embed, y
        # return x_previous, x_upcoming, y
        return x, y
