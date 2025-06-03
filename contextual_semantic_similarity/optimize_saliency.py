from scipy.optimize import minimize
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.functional import cross_entropy
import pandas as pd
import time
import torch
import random
from collections import defaultdict
from itertools import combinations
from scipy.stats import ttest_rel

from contextual_semantic_similarity.visualizations import display_prediction_distribution
from prepare_data import (convert_data_to_tensors, split_data, compute_split_arrays,
                          compute_participant_indices, compute_participant_split_array,
                          load_baseline_tensors, clean_tensors)
from visualizations import display_error

def normalize_true_pred(true_targets, pred_targets, method='min-max'):

    if method == 'z-score':
        true_targets = true_targets.float()
        pred_targets = pred_targets.float()
        true_targets = (true_targets - torch.mean(true_targets))/torch.std(true_targets)
        pred_targets = (pred_targets - torch.mean(pred_targets))/torch.std(pred_targets)
        # TODO account for division by 0
    else:
        max_pos = torch.max(true_targets)  # 3 if word level, letter varies with dataset e.g. 39
        min_pos = torch.min(true_targets)  # -3 if word level, letter varies with dataset e.g. -36
        true_targets = (true_targets - min_pos) / (max_pos - min_pos)
        pred_targets = (pred_targets - min_pos) / (max_pos - min_pos)

    return true_targets, pred_targets

def predict(x, input_features, device, n_features:int=5, context_window_size:int=7, mapping_type:str='dist_max', level_type:str='word', letter_positions=None):

    # position: weight = {-3: .25, -2: .50, -1: .75, 0: .75, 1: 1, 2: .75, 3: .5}
    distance_weights = torch.tensor([.25, .50, .75, .75, 1, .75, .5]).to(device)

    if level_type == 'letter' and letter_positions == None:
        raise ValueError('If level is letter, letter positions must be provided.')

    # print(input_features.shape)
    # print(input_features[0])
    # multiply features with weights
    weights = torch.zeros((n_features, context_window_size)).to(device)
    for i, weight in enumerate(x):
        weights[i][:] = weight
    weighted_input = input_features * weights[None,:,:]
    # print(weights[0])
    # print(weighted_input[0])

    # sum features to get score for each context word
    # from (n_fixations, n_features, context_window_size) to (n_fixations, context_window_size)
    weighted_input = torch.sum(weighted_input, dim=-2)
    # print(weighted_input[0])

    # multiply score with distance weight if mapping_type='dist_max'
    if 'dist' in mapping_type:
        assert distance_weights.shape == weighted_input[0].shape
        weighted_input = weighted_input * distance_weights[None,:]

    # map saliency scores onto target prediction, from (n_fixations, context_window_size) to (n_fixations)
    if mapping_type == 'centre_mass':
        # with positions shifted to the right to simulate focus of attention shifted to the right
        # TODO should positions here also be normalized?
        if level_type == 'letter':
            positions = letter_positions.to(device)
            pred_targets = torch.sum(weighted_input * positions, dim=-1)/torch.sum(weighted_input,dim=-1)
        else: # level_type = word
            positions = torch.tensor([-4,-3,-2,-1,0,1,2]).to(device)
            pred_targets = torch.sum(weighted_input * positions[None,:], dim=-1)/torch.sum(weighted_input,dim=-1)
    # elif 'score' in mapping_type: # treating the problem as classification, with context window positions as the classes
    #     pass # TODO check if scores are between 0 and 1
    else: # dist_max or raw_max
        if level_type == 'letter':
            pred_targets = torch.argmax(weighted_input, dim=-1, keepdim=True)
            pred_targets = torch.gather(letter_positions, dim=-1, index=pred_targets).squeeze()
        else: # level_type = word
            pred_targets = torch.argmax(weighted_input, dim=-1)
            # scale indices to -3+3 range
            scale = torch.tensor([-3]).to(device)
            pred_targets = pred_targets + scale

    return pred_targets

def compute_error(true, pred, loss_function='mse'):

    if loss_function == 'mse':
        return mean_squared_error(true, pred)

    elif loss_function == 'rmse':
        return np.sqrt(mean_squared_error(true, pred))

    elif loss_function == 'mae':
        return mean_absolute_error(true, pred)

    elif loss_function == 'log_loss':
        return cross_entropy(true,pred)

def objective(x, input_features, true_targets, device, n_features:int=5, context_window_size:int=7, mapping_type:str='dist_max', level_type:str='word', letter_positions=None, loss_function='mse'):

    error = None

    # if not all weights 0
    if not all(v == 0. for v in x):

        pred_targets = predict(x=x, input_features=input_features,
                                device=device, n_features=n_features,
                                context_window_size=context_window_size, mapping_type=mapping_type,
                                level_type=level_type, letter_positions=letter_positions)
        assert pred_targets.shape == true_targets.shape
        # print(true_targets.shape, pred_targets.shape)
        # remove possible NaN values
        true_targets, pred_targets = clean_tensors(true_targets, pred_targets)
        # print(true_targets.shape, pred_targets.shape)
        # normalize predicted and true targets
        true_targets, pred_targets = normalize_true_pred(true_targets, pred_targets)
        # print(pred_targets[0])

        error = compute_error(true_targets, pred_targets, loss_function)

    return error

def optimize(input_features, true_targets, x_prior, device, n_features:int=5, context_window_size:int=7, mapping_type='raw_max', level_type='word', letter_positions=None, loss_function='mse'):

    print('Optimizing...')
    initial_error, final_error, change_error = None, None, None
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    solution = minimize(fun=objective,
                        x0=x_prior,
                        args=(input_features, true_targets, device, n_features, context_window_size, mapping_type, level_type, letter_positions, loss_function),
                        method='Nelder-Mead', # Nelder-Mead # BFGS
                        #jac='3-point',
                        options={'disp': True})
    print('Success: ', solution.success)
    if not solution.success:
        print('Message: ', solution.message)
    else:
        print('Solution: ', solution.x)
        initial_error = objective(x_prior, input_features, true_targets, device, n_features, context_window_size, mapping_type, level_type, letter_positions, loss_function)
        final_error = solution.fun
        change_error = -((initial_error - final_error)/initial_error)
        print('Initial error: ', initial_error)
        print('Final error: ', final_error)
        print('Change in error: ', change_error)

    return solution.x, initial_error, final_error, change_error

def save_train_results(split, weights, initial_error, final_error, change_error, time_elapsed, weights_filepath):

    mode = 'a'
    if split == 0:
        mode = 'w'

    with open(weights_filepath, mode) as f:
        if split == 0:
            f.write('train_split\tweights\tinitial_error\tfinal_error\tchange_error\ttime_elapsed\n')
        f.write(f'{split}\t{np.array2string(weights)}\t{initial_error}\t{final_error}\t{change_error}\t{time_elapsed/60}\n')

def save_split_results(split, error, filepath):

    mode = 'a'
    if split == 0:
        mode = 'w'

    with open(filepath, mode) as f:
        if split == 0:
            f.write('split\terror\n')
        f.write(f'{split}\t{error}\n')

def save_baseline_results(split, error, error_change, filepath):

    mode = 'a'
    if split == 0:
        mode = 'w'

    with open(filepath, mode) as f:
        if split == 0:
            f.write('split\terror\tbaseline-model\n')
        f.write(f'{split}\t{error}\t{error_change}\n')

def test_sig_diff(all_errors, all_conditions, opt_dir, loss_function='mse'):

    df = pd.DataFrame({'error': all_errors, 'condition': all_conditions})
    error_dict = defaultdict(list)
    for condition, errors in df.groupby('condition'):
        error_dict[condition] = errors['error'].tolist()
    for condition_combi in combinations(df['condition'].unique().tolist(), 2):
        result = ttest_rel(error_dict[condition_combi[0]], error_dict[condition_combi[1]])
        with open(f'{opt_dir}/t-test_{condition_combi}_{loss_function}.csv', 'w') as f:
            f.write('t-statistic\tp-value\tdf\n')
            f.write(f'{result.statistic}\t{result.pvalue}\t{result.df}\n')

def main():

    model_name = 'gpt2'
    layers = '11'
    corpus_name = 'meco'
    eye_data_filepath = f'data/processed/{corpus_name}/{model_name}/full_{model_name}_[{layers}]_{corpus_name}_window_cleaned.csv'
    words_filepath = f'data/processed/{corpus_name}/words_en_df.csv'
    mappings = 'dist_max,centre_mass' # dist_max, raw_max, centre_mass
    level = 'word' # letter, word
    compute_arrays = False
    pre_process = False
    opt_dir = f'data/processed/{corpus_name}/{model_name}/optimization'
    loss_function = 'mae'
    n_features = 4 # similarity, length, entropy, surprisal

    eye_data = pd.read_csv(eye_data_filepath)
    words_data = pd.read_csv(words_filepath)

    if compute_arrays:
        convert_data_to_tensors(eye_data, words_data, opt_dir, level, pre_process=pre_process)

    # determine device to run optimization
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('Device: ', device)

    # setting seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # find split indices
    print('Splitting data into train and test...')
    split_indices_test = split_data(eye_data['trialid'].unique(), split_type='train-test', test_size=.1, shuffle=True,
                                    random_state=42,
                                    filepath=f'{opt_dir}/train_test_split_opt.txt')
    train_eye_data = eye_data[eye_data['trialid'].isin(split_indices_test[0]['train_index'])].copy()
    print('Splitting train data for cross-validation of optimizer...')
    # create folds with remaining data for cross-validation
    split_indices = split_data(train_eye_data['trialid'].unique(), n_splits=5, shuffle=True, random_state=42,
                               filepath=f'{opt_dir}/cross_val_splits_opt.txt')


    start_time = time.perf_counter()

    all_errors, all_conditions = [],[] # to display errors
    all_targets, all_predictions, all_models = [], [], [] # to display prediction distributions

    # run optimization on each training set
    for i, split in enumerate(split_indices):
        print('Split ', i)

        input_features, true_targets, letter_pos = compute_split_arrays(directory=opt_dir, level=level,
                                                                        trialids=split['train_index'],
                                                                        concatenate=False, class_indices=False)
        test_input_features, test_true_targets, test_letter_pos = compute_split_arrays(directory=opt_dir, level=level,
                                                                                        trialids=split['test_index'],
                                                                                        concatenate=False, class_indices=False)

        for mapping in mappings.split(','):

            print(f'Optimizing saliency formula with mapping {mapping} at {level} level...')
            weights, initial_error, final_error, change_error = optimize(input_features=input_features,
                                                                         true_targets=true_targets,
                                                                         x_prior=[-.1,.1,.1,.1],
                                                                         device=device,
                                                                         mapping_type=mapping, level_type=level,
                                                                         letter_positions=letter_pos,
                                                                         loss_function=loss_function,
                                                                         n_features=n_features)
            time_elapsed = time.perf_counter() - start_time
            print("Time elapsed: " + str(time_elapsed/60) + " minutes")
            weights_filepath = f'{opt_dir}/saliency_{mapping}_{level}_optimized_weights.txt'
            save_train_results(i, weights, initial_error, final_error, change_error, time_elapsed, weights_filepath)

            # compute error on test split
            print("Testing optimized weights...")
            pred_targets = predict(x=weights, input_features=test_input_features, device=device, mapping_type=mapping,
                                   level_type=level, letter_positions=test_letter_pos, n_features=n_features)
            clean_true_targets, clean_pred_targets = clean_tensors(test_true_targets, pred_targets)
            norm_true_targets, norm_pred_targets = normalize_true_pred(clean_true_targets, clean_pred_targets)
            test_error = compute_error(norm_true_targets, norm_pred_targets, loss_function=loss_function)
            save_split_results(i, test_error, filepath = f'{opt_dir}/saliency_{mapping}_{level}_test.txt')
            all_errors.append(test_error)
            all_conditions.append(mapping)
            all_predictions.extend(clean_pred_targets)
            all_targets.extend(clean_true_targets)
            all_models.extend([mapping for i in clean_pred_targets])

        # compare it with baseline
        for baseline in ['next_word']:
            base_pred_targets = load_baseline_tensors(split['test_index'], baseline, opt_dir, level)
            clean_test_true_targets, clean_base_pred_targets = clean_tensors(test_true_targets, base_pred_targets)
            norm_test_true_targets, norm_base_pred_targets = normalize_true_pred(clean_test_true_targets,
                                                                                 clean_base_pred_targets)
            base_error = compute_error(norm_test_true_targets, norm_base_pred_targets, loss_function=loss_function)
            all_errors.append(base_error)
            all_conditions.append(baseline)
            all_predictions.extend(clean_base_pred_targets)
            all_targets.extend(clean_test_true_targets)
            all_models.extend([baseline for i in clean_base_pred_targets])


    display_error(all_errors, all_conditions, loss_function, f'{opt_dir}/error_{mappings}_{level}.tiff')
    test_sig_diff(all_errors, all_conditions, opt_dir, loss_function)
    display_prediction_distribution(all_targets, all_predictions, f'{opt_dir}/distribution_opt_{mappings}_{level}.tiff', col=all_models)

    # ---------------------------------------------------

    start_time = time.perf_counter()

    all_participants, all_predictions, all_targets, all_conditions = [], [], [], []  # to display pred distribution graph
    all_error_participants, all_errors, all_error_conditions = [], [], []  # to display error graph

    # sample participants
    p = eye_data['participant_id'].unique().tolist()
    # participant did not read all texts
    p.remove('en_21')
    p.remove('en_47')
    p.remove('en_49')
    participant_set = random.sample(p, 6)

    # Per participant
    print('Optimizing saliency per participant...')
    for i, split in enumerate(split_indices):

        print('Split ', i)

        input_features, true_targets, letter_pos = compute_split_arrays(trialids=split['train_index'],
                                                                        directory=opt_dir, level=level,
                                                                        concatenate=False,
                                                                        class_indices=False)
        test_input_features, test_true_targets, test_letter_pos = compute_split_arrays(directory=opt_dir, level=level,
                                                                                       trialids=split['test_index'],
                                                                                       concatenate=False,
                                                                                       class_indices=False)

        participant_indices = compute_participant_indices(eye_data, split['train_index'])
        participant_test_indices = compute_participant_indices(eye_data, split['test_index'])

        for participant, participant_idx_list in participant_indices.items():

            if participant in participant_set:

                print('Participant ', participant)

                p_input_features, p_true_targets, p_letter_pos = compute_participant_split_array(participant_idx_list,
                                                                                                 input_features,
                                                                                                 true_targets,
                                                                                                 letter_pos)
                p_test_input_features, p_test_true_targets, p_test_letter_pos = compute_participant_split_array(
                                                                                participant_test_indices[participant],
                                                                                test_input_features,
                                                                                test_true_targets,
                                                                                test_letter_pos)


                for mapping in mappings.split(','):

                    print(f'Optimizing saliency formula with mapping {mapping} at {level} level...')

                    weights, initial_error, final_error, change_error = optimize(input_features=p_input_features,
                                                                                 true_targets=p_true_targets,
                                                                                 x_prior=[-.1, .1, .1, .1],
                                                                                 device=device,
                                                                                 mapping_type=mapping, level_type=level,
                                                                                 letter_positions=p_letter_pos,
                                                                                 loss_function=loss_function,
                                                                                 n_features=n_features)
                    time_elapsed = time.perf_counter() - start_time
                    print("Time elapsed: " + str(time_elapsed / 60) + " minutes")

                    weights_filepath = f'{opt_dir}/saliency_{mapping}_{level}_{participant}_optimized_weights.txt'
                    save_train_results(i, weights, initial_error, final_error, change_error, time_elapsed, weights_filepath)

                    # compute error on test split
                    print("Testing optimized weights...")
                    p_pred_targets = predict(x=weights, input_features=p_test_input_features, device=device, mapping_type=mapping,
                                           level_type=level, letter_positions=p_test_letter_pos, n_features=n_features)
                    # clean targets and save them to display distributions
                    clean_p_true_targets, clean_p_pred_targets = clean_tensors(p_test_true_targets, p_pred_targets)
                    all_predictions.extend(clean_p_pred_targets)
                    all_targets.extend(clean_p_true_targets)
                    all_participants.extend([participant for i in clean_p_pred_targets])
                    all_conditions.extend([mapping for i in clean_p_pred_targets])
                    # compute error and save them to display error
                    norm_true_targets, norm_pred_targets = normalize_true_pred(clean_p_true_targets, clean_p_pred_targets)
                    test_error = compute_error(norm_true_targets, norm_pred_targets, loss_function=loss_function)
                    save_split_results(i, test_error, filepath=f'{opt_dir}/saliency_{mapping}_{level}_{participant}_test.txt')
                    all_errors.append(test_error)
                    all_error_conditions.append(mapping)
                    all_error_participants.append(participant)

                # compute baseline errors for each participant
                for baseline in ['next_word']:
                    base_pred_targets = load_baseline_tensors(split['test_index'], baseline, opt_dir, level)
                    participant_idx_tensor = torch.tensor(participant_test_indices[participant])
                    p_base_pred_targets = torch.index_select(base_pred_targets, dim=0, index=participant_idx_tensor)
                    p_test_true_targets, p_base_pred_targets = clean_tensors(p_test_true_targets, p_base_pred_targets)
                    norm_test_true_targets, norm_base_pred_targets = normalize_true_pred(p_test_true_targets,
                                                                                         p_base_pred_targets)
                    base_error = compute_error(norm_test_true_targets, norm_base_pred_targets,
                                              loss_function=loss_function)
                    all_errors.append(base_error)
                    all_error_conditions.append(baseline)
                    all_error_participants.append(participant)

    # display errors
    display_error(all_errors, all_error_conditions, loss_function,
                  f"{opt_dir}/error_{mappings}_{level}_{participant_set}.tiff",
                  col=all_error_participants,
                  col_name='participant')
    # display prediction distributions
    df = pd.DataFrame({'participant': all_participants,
                       'condition': all_conditions,
                       'target': all_targets,
                       'prediction': all_predictions})
    for condition, rows in df.groupby('condition'):
        display_prediction_distribution(rows['target'].tolist(), rows['prediction'].tolist(),
                                        f"{opt_dir}/distribution_opt_{condition}_{level}_{participant_set}.tiff",
                                        col=rows['participant'].tolist())


if __name__ == '__main__':
    main()