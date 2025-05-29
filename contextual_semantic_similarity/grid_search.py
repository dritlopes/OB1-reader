import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import product
import time
from prepare_data import convert_data_to_tensors, split_data, compute_split_arrays, clean_tensors
from optimize_saliency import predict, normalize_true_pred, compute_error, objective
from visualizations import display_prediction_distribution

def log_sample(start, end, num, base):
    if not base: base = 10.0
    return np.logspace(start, end, num, base=base)

def line_sample(start, end, num):
    return np.linspace(start, end, num)

def grid_search(x:list[float], input_features, true_targets, mapping, level, letter_positions, device, n_features:int=5, context_window_size:int=7, topk:int=5, sampling_method=None, base=None, display=False, filepath='', loss_function='mse', verbose=False):

    parameter_scoring = {'parameters':[],
                         'rmse': []}

    if sampling_method == 'log':
        values = np.round(log_sample(x[0], x[1], n_features, base),2)
        if display:
            plt.scatter(values, y = np.ones(n_features), color='green')
            plt.xticks(values)
            plt.title('logarithmically spaced numbers')
            plt.show()
    elif sampling_method == 'line':
        values = np.round(line_sample(x[0], x[1], n_features), 2)
        if display:
            plt.scatter(values, y=np.ones(n_features), color='green')
            plt.xticks(values)
            plt.title('linearly spaced numbers')
            plt.show()
    elif not sampling_method:
        values = x
    else:
        raise ValueError('Sampling method not supported.')

    for combination in product(values, repeat=n_features):
        start_time = time.perf_counter()
        rmse_i = objective(x=combination,
                           input_features=input_features,
                           true_targets=true_targets,
                           device=device,
                           n_features=n_features,
                           context_window_size=context_window_size,
                           mapping_type=mapping,
                           level_type=level,
                           letter_positions=letter_positions,
                           loss_function=loss_function)
        parameter_scoring['parameters'].append(combination)
        parameter_scoring['rmse'].append(rmse_i)
        time_elapsed = time.perf_counter() - start_time
        if verbose:
            print('Weights: ', combination)
            print('RMSE: ', rmse_i)
            print("Time elapsed: " + str(time_elapsed / 60) + " minutes")

    df = pd.DataFrame.from_dict(parameter_scoring)
    df.sort_values(by=['rmse'], ascending=True, inplace=True)
    if filepath:
        df.to_csv(filepath, index=False)

    best_parameters = df['parameters'].tolist()[:topk]

    return best_parameters

def main():

    model_name = 'gpt2'
    layers = '11'
    corpus_name = 'meco'
    eye_data_filepath = f'data/processed/{corpus_name}/{model_name}/full_{model_name}_[{layers}]_{corpus_name}_window_cleaned.csv'
    words_filepath = f'data/processed/{corpus_name}/words_en_df.csv'
    mapping = 'dist_max'  # dist_max, raw_max, centre_mass
    level = 'word'  # letter, word
    compute_arrays = False
    pre_process = False
    opt_dir = f'data/processed/{corpus_name}/{model_name}/optimization'
    loss_function = 'mae'

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

    print('Splitting train data for grid-search...')
    # split remaining data into train and val for grid-search
    split_indices_grid_search = split_data(eye_data['trialid'].unique(), split_type='train-test', test_size=.2,
                                           shuffle=True,
                                           random_state=42,
                                           filepath=f'{opt_dir}/grid_search_split.txt')

    # do grid-search to find prior for optimization
    print('Do grid-search to define initial weights...')
    i_train = split_indices_grid_search[0]['train_index']
    input_features, true_targets, letter_pos = compute_split_arrays(directory=opt_dir, level=level,
                                                                    split_trialids=i_train, device=device,
                                                                    verbose=False)
    grid_search_filepath = f'{opt_dir}/grid_search_{level}_{mapping}.csv'
    best_parameters = grid_search(x=[-.4, -.2, 0., .2, .4, .6, .8, 1.], input_features=input_features, true_targets=true_targets,
                              mapping=mapping, level=level, letter_positions=letter_pos, device=device,
                              filepath=grid_search_filepath, loss_function=loss_function, verbose=False)
    # best_parameters = [[1.0, -0.4, 1.0, 1.0, 1.0]]
    # evaluate top-k parameters on validation set
    print('Validating grid-search...')
    i_val = split_indices_grid_search[0]['test_index']
    input_features, true_targets, letter_pos = compute_split_arrays(directory=opt_dir, level=level,
                                                                    split_trialids=i_val, device=device, verbose=False)
    val_rmse, val_parameters = [],[]
    for i, parameters in enumerate(best_parameters):
        pred_targets = predict(x=parameters, input_features=input_features, device=device, mapping_type=mapping,
                               level_type=level, letter_positions=letter_pos)
        true_targets, pred_targets = clean_tensors(true_targets, pred_targets)
        norm_true_targets, norm_pred_targets = normalize_true_pred(true_targets, pred_targets)
        error = compute_error(norm_true_targets, norm_pred_targets, loss_function=loss_function)
        display_prediction_distribution(true_targets, pred_targets, f'{opt_dir}/distribution_{mapping}_{level}_{parameters}.tiff')
        grid_search_val_filepath = f'{opt_dir}/grid_search_val_{level}_{mapping}.csv'
        val_parameters.append(parameters)
        val_rmse.append(error)
        pd.DataFrame({'parameters': val_parameters, 'error': val_rmse}).to_csv(grid_search_val_filepath, index=False)
        print(f'Error of top {i+1} initial weights on validation set: {error}')

    # per participant
    # print('Do grid-search per participant to define initial weights...')
    # i_train = split_indices_grid_search[0]['train_index']
    # input_features, true_targets, letter_pos = compute_split_arrays(directory=opt_dir, level=level,
    #                                                                 split_trialids=i_train, device=device,
    #                                                                 verbose=False)
    # participant_indices = compute_participant_indices(eye_data, i_train)
    # i_val = split_indices_grid_search[0]['test_index']
    # val_input_features, val_true_targets, val_letter_pos = compute_split_arrays(directory=opt_dir, level=level,
    #                                                                             split_trialids=i_val, device=device,
    #                                                                             verbose=False)
    # participant_indices_val = compute_participant_indices(eye_data, i_val)
    # for participant, participant_idx_list in participant_indices.items():
    #     if participant in ['en_10', 'en_101', 'en_102', 'en_11', 'en_14']:
    #         p_input_features, p_true_targets, p_letter_pos = compute_participant_split_array(participant_idx_list,
    #                                                                                          input_features,
    #                                                                                          true_targets,
    #                                                                                          letter_pos)
    #         grid_search_filepath = f'{opt_dir}/grid_search_{level}_{mapping}_{participant}.csv'
    #         best_parameters = grid_search(x=[-.4, -.2, 0., .2, .4, .6, .8, 1.], input_features=p_input_features,
    #                                       true_targets=p_true_targets, mapping=mapping, level=level,
    #                                       letter_positions=p_letter_pos, device=device,
    #                                       filepath=grid_search_filepath, loss_function='rmse', verbose=False)
    #         print('Validating grid-search...')
    #         p_val_input_features, p_val_true_targets, p_val_letter_pos = compute_participant_split_array(
    #             participant_indices_val[participant],
    #             val_input_features, val_true_targets,
    #             val_letter_pos)
    #         val_rmse, val_parameters = [], []
    #         for i, parameters in enumerate(best_parameters):
    #             p_val_pred_targets = predict(x=parameters, input_features=p_val_input_features,
    #                                          device=device, mapping_type=mapping,
    #                                          level_type=level, letter_positions=p_val_letter_pos)
    #             p_val_true_targets, p_val_pred_targets = clean_tensors(p_val_true_targets, p_val_pred_targets)
    #             norm_true_targets, norm_pred_targets = normalize_true_pred(p_val_true_targets, p_val_pred_targets)
    #             error = compute_error(norm_true_targets, norm_pred_targets, loss_function='rmse')
    #             check_prediction_distribution(p_val_true_targets, p_val_pred_targets,
    #                                           f'{opt_dir}/distribution_{mapping}_{level}_{parameters}_{participant}.tiff')
    #             grid_search_val_filepath = f'{opt_dir}/grid_search_val_{level}_{mapping}_{participant}.csv'
    #             val_parameters.append(parameters)
    #             val_rmse.append(error)
    #             pd.DataFrame({'parameters': val_parameters, 'error': val_rmse}).to_csv(grid_search_val_filepath,
    #                                                                                    index=False)
    #             print(f'Error of top {i + 1} initial weights on validation set: {error}')