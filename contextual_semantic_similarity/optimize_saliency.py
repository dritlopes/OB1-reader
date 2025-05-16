from scipy.optimize import minimize
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from compute_saliency import find_letter_distance_2centre_of_context_word, find_letter_distances, compute_letter_map, normalize, baseline_7letter_2right
from sklearn.model_selection import KFold
import time
import math
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split
import scipy

def split_data(df, split_type='cross-validation', n_splits=5, test_size=.2, shuffle=False, random_state=42, filepath=''):

    splits = []

    if split_type == 'cross-validation':
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_index, test_index in kf.split(df['trialid'].unique().tolist()):
            splits.append({'train_index': train_index, 'test_index': test_index})
            # print('train_index', train_index)
            # print('test_index', test_index)
    else:
        train, test = train_test_split(df['trialid'].unique().tolist(), test_size=test_size, shuffle=shuffle, random_state=random_state)
        splits.append({'train_index': train, 'test_index': test})

    if filepath:
        with open(filepath, 'w') as f:
            f.write('split\ttrain\ttest\n')
            for i, split in enumerate(splits):
                f.write(f'{i}\t{split["train_index"]}\t{split["test_index"]}\n')

    return splits

def predict(x, fixation_data:pd.DataFrame, mapping_type='raw_max', level_type='word', letter_map=None):

    end_positions, pred_end_positions = [], []
    distance_weights = {-3: .25, -2: .50, -1: .75, 0: .75, 1: 1, 2: .75, 3: .5}

    if level_type == 'letter' and not letter_map:
        raise ValueError('If level is letter, letter_map must be provided.')

    for i, context in fixation_data.groupby(['participant_id', 'trialid', 'fixid']):
        end_position = None
        for context_word in context.itertuples():
            # register true landing position
            if context_word.landing_target:
                if level_type == 'letter':
                    end_position = context_word.letter_distance
                else:  # level_type = word
                    end_position = context_word.distance

        # find which word and letter 7 letters to the right
        pos_letter7_2right = baseline_7letter_2right(context, letter_map, level_type='word')

        # only if end_position (for a few instances, end position was deleted due to one of the word features being none.)
        if end_position != None:
            end_positions.append(end_position)
            scores = []
            for context_word in context.itertuples():
                letter7_2right = 0.0
                if context_word.distance == pos_letter7_2right:
                    letter7_2right = 1.0
                # compute saliency scores
                score = (((1-context_word.similarity) * x[0]) # semantic distance
                         + (context_word.norm_length * x[1])
                         + (context_word.norm_surprisal * x[2])
                         + (context_word.norm_entropy * x[3])
                         + (letter7_2right * x[4]))
                if mapping_type == 'dist_max':
                    score = score * distance_weights[context_word.distance]
                scores.append(score)
            # predict next eye movement landing target
            if mapping_type == 'centre_mass':
                if level_type == 'letter':
                    # find letter distances to fixation
                    positions = find_letter_distances(context, letter_map)
                else:  # level_type = word
                    positions = [position - 1 for position in context['distance'].tolist()]
                pred_position = (1 / np.sum(scores)) * np.sum(
                    [saliency * position for saliency, position in zip(scores, positions)])
                pred_end_positions.append(pred_position)
            else:  # winner-takes-all (maximum score)
                winner_word = context.iloc[scores.index(max(scores))]
                if level_type == 'word':
                    pred_end_positions.append((winner_word['distance']))
                elif level_type == 'letter':
                    pred_end_positions.append(find_letter_distance_2centre_of_context_word(winner_word, letter_map))

    # normalize predicted and true positions
    max_pos = max([pos for pos in end_positions if not math.isnan(pos)]) # 3 if word level, 39 or 37 if letter
    min_pos = min([pos for pos in end_positions if not math.isnan(pos)]) # -3 if word level,
    end_positions = [(pos - min_pos) / (max_pos - min_pos) for pos in end_positions]
    pred_end_positions = [(pos - min_pos) / (max_pos - min_pos) for pos in pred_end_positions]

    return end_positions, pred_end_positions

def objective(x, fixation_data:pd.DataFrame, mapping_type='raw_max', level_type='word', letter_map=None):

    end_positions, pred_end_positions = predict(x, fixation_data, mapping_type, level_type, letter_map)
    # compute rmse
    rmse = round(np.sqrt(mean_squared_error(end_positions, pred_end_positions)), 2)

    return rmse

def log_sample(start, end, num, base):
    if not base: base = 10.0
    return np.logspace(start, end, num, base=base)

def line_sample(start, end, num):
    return np.linspace(start, end, num)

def grid_search(x:list[float], df, mapping, level, letter_map, n:int=5, sampling_method=None, base=None, display=False, filepath='', verbose=False):

    parameter_scoring = {'parameters':[],
                         'rmse': []}

    if sampling_method == 'log':
        values = np.round(log_sample(x[0], x[1], n, base),2)
        if display:
            plt.scatter(values, y = np.ones(n), color='green')
            plt.xticks(values)
            plt.title('logarithmically spaced numbers')
            plt.show()
    elif sampling_method == 'line':
        values = np.round(line_sample(x[0], x[1], n), 2)
        if display:
            plt.scatter(values, y=np.ones(n), color='green')
            plt.xticks(values)
            plt.title('linearly spaced numbers')
            plt.show()
    elif not sampling_method:
        values = x
    else:
        raise ValueError('Sampling method not supported.')

    for combination in product(values, repeat=5):
        start_time = time.perf_counter()
        x_i = np.array(combination)
        rmse_i = objective(x_i, df, mapping, level, letter_map)
        parameter_scoring['parameters'].append(x_i)
        parameter_scoring['rmse'].append(rmse_i)
        time_elapsed = time.perf_counter() - start_time
        if verbose:
            print('Weights: ', x_i)
            print('RMSE: ', rmse_i)
            print("Time elapsed: " + str(time_elapsed / 60) + " minutes")

    if filepath:
        df = pd.DataFrame.from_dict(parameter_scoring)
        df.to_csv(filepath, index=False)

    best_parameters = parameter_scoring['parameters'][parameter_scoring['rmse'].index(min(parameter_scoring['rmse']))]

    return best_parameters

def optimize(fixation_data:pd.DataFrame, x_prior, mapping_type='raw_max', level_type='word', letter_map=None):

    print('Optimizing...')
    initial_error, final_error, change_error = None, None, None
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    solution = minimize(fun=objective,
                        x0=x_prior,
                        args=(fixation_data, mapping_type, level_type, letter_map),
                        method='BFGS', # Nelder-Mead # BFGS
                        jac='3-point',
                        options={'disp': True})
    print('Success: ', solution.success)
    if not solution.success:
        print('Message: ', solution.message)
    else:
        print('Solution: ', solution.x)
        initial_error = objective(np.array([.01, .01, .01, .01, .01]),
                                  fixation_data, mapping_type, level_type, letter_map)
        final_error = solution.fun
        change_error = -((initial_error - final_error)/initial_error)
        print('Initial error: ', initial_error)
        print('Final error: ', final_error)
        print('Change in error: ', change_error)

    return solution.x, initial_error, final_error, change_error

def save_train_results(split, weights, initial_error, final_error, change_error, time_elapsed, weights_filepath):

    with open(weights_filepath, 'a') as f:
        if split == 0:
            f.write('train_split\tweights\tinitial_error\tfinal_error\tchange_error\ttime_elapsed\n')
        f.write(f'{split}\t{np.array2string(weights)}\t{initial_error}\t{final_error}\t{change_error}\t{time_elapsed/60}\n')

def main():

    model_name = 'gpt2'
    layers = '11'
    corpus_name = 'meco'
    eye_data_filepath = f'data/processed/{corpus_name}/{model_name}/full_{model_name}_[{layers}]_{corpus_name}_window_df.csv'
    words_filepath = f'data/processed/{corpus_name}/words_en_df.csv'
    mapping_type = ['dist_max', 'centre_mass', 'raw_max']
    level_type = ['word', 'letter']

    eye_data = pd.read_csv(eye_data_filepath, index_col=0)
    words_data = pd.read_csv(words_filepath, index_col=0)

    # remove nan values
    eye_data.dropna(subset=["similarity","length","entropy","surprisal"], inplace=True)

    # find split indices
    print('Splitting data into train and test...')
    split_indices_test = split_data(eye_data, split_type='train-test', test_size=.1, shuffle=True, random_state=42,
                                           filepath=f'data/processed/{corpus_name}/{model_name}/optimization/train_test_split.txt')
    train_eye_data = eye_data[eye_data['trialid'].isin(split_indices_test[0]['train_index'])].copy()
    print('Splitting train data for grid-search...')
    # split remaining data into train and val for grid-search
    split_indices_grid_search = split_data(train_eye_data, split_type='train-test', test_size=.2, shuffle = True,
                                           random_state = 42,
                                           filepath = f'data/processed/{corpus_name}/{model_name}/optimization/grid_search_split.txt')
    print('Splitting train data for cross-validation of optimizer...')
    # create folds with remaining data for cross-validation
    split_indices = split_data(train_eye_data, n_splits=5, shuffle=True, random_state=42,
                               filepath=f'data/processed/{corpus_name}/{model_name}/optimization/cross_val_splits.txt')

    # find index of all letters in each word in each text
    print('Computing letter distances...')
    letter_map = compute_letter_map(words_data)

    # normalize variables for combination computation
    print('Normalizing features...')
    for feature in ['length', 'entropy', 'surprisal']:
        norm_feature = normalize(eye_data[feature].tolist())
        eye_data[f'norm_{feature}'] = norm_feature
    # convert entropy values from previous context to 0.0001
    eye_data['norm_entropy'] = eye_data.apply(
        lambda x: 0.0001 if x['distance'] in [-3, -2, -1, 0] else x['norm_entropy'], axis=1)

    # run optimizer on training sets
    start_time = time.perf_counter()
    for level in level_type:
        for mapping in mapping_type:

            # do grid-search to find prior for optimization
            print('Do grid-search to define initial weights...')
            grid_search_filepath = f'data/processed/{corpus_name}/{model_name}/optimization/grid_search_{level}_{mapping}.txt'
            i_train = split_indices_grid_search[0]['train_index']
            eye_data_train = eye_data[eye_data['trialid'].isin(i_train)].copy()
            best_parameters = grid_search([-.25, 0., .25, .75], eye_data_train, mapping, level, letter_map,
                                          filepath=grid_search_filepath, verbose=True)
            # evaluate parameters on validation set
            i_val = split_indices_grid_search[0]['test_index']
            eye_data_val = eye_data[eye_data['trialid'].isin(i_val)].copy()
            val_rmse = objective(best_parameters, eye_data_val, mapping, level, letter_map)
            print('RMSE of best initial weights on validation set: ', val_rmse)

            # # run optimization on each training set
            # for i, split in enumerate(split_indices):
            #     print(f'Optimizing saliency formula with mapping {mapping} at {level} level...')
            #     print('Training split ', i)
            #     eye_data_filtered = eye_data[eye_data['trialid'].isin(split['train_index'])].copy()
            #     # eye_data_filtered = eye_data[eye_data['trialid'] == 0].copy()
            #     weights, initial_error, final_error, change_error = optimize(eye_data_filtered, best_parameters, mapping, level, letter_map)
            #     time_elapsed = time.perf_counter() - start_time
            #     print("Time elapsed: " + str(time_elapsed/60) + " minutes")
            #     weights_filepath = f'data/processed/{corpus_name}/{model_name}/optimization/saliency_{mapping}_{level}_split{i}_optimized_weights.txt'
            #     save_train_results(i, weights, initial_error, final_error, change_error, time_elapsed, weights_filepath)
            #     # TODO compute RMSE on test split
    # opt on first text works with letter level and centre of mass (method BFSG, jac 3-point); dist_max letter does not work;
    # but not with word level centre of mass, nor dist_max (method BFSG, jac 3-point,cs, smaller tol neither); split 1 same thing;
    # let's do grid search to define the initial guess and see if improving the initial guess gives us a better result

if __name__ == '__main__':
    main()