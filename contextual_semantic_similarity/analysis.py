from scipy import stats
import pandas as pd
import seaborn as sb
import numpy as np
import os
import matplotlib.pyplot as plt

def correlate_eye2sim(layers, paths_to_data, measures, model_name, dir_to_save):

    print('Measuring correlation between similarity and eye movements...')

    all_measures, all_coef, all_layers = [], [], []

    for layer_combi, layer_combi_data in zip(layers, paths_to_data):

        all_corr = []

        eye_move_sim_df = pd.read_csv(layer_combi_data)
        measures = [measure for measure in measures if measure in eye_move_sim_df.columns]

        for measure in measures:
            clean_df = eye_move_sim_df.dropna(subset=[measure, 'similarity'])
            # take the mean of each eye mov measure for each word
            mean_measure_df = clean_df.groupby(['trialid', 'ianum'], as_index=False)[measure].mean()
            # take the mean contextual similarity for each word
            mean_similarity_df = clean_df.groupby(['trialid', 'ianum'], as_index=False)['similarity'].mean()
            # measure correlation between the two
            cor = stats.pearsonr(mean_measure_df[measure].tolist(), mean_similarity_df['similarity'].tolist())
            all_corr.append(cor)

        # save the correlation values for each layer similarity
        cor_df = pd.DataFrame({'measure': measures,
                               'coefficient': [cor.statistic for cor in all_corr],
                               'p-value': [cor.pvalue for cor in all_corr]})
        cor_df.to_csv(f'{dir_to_save}/sim_{layer_combi}_{model_name}_pearsonr_corr.csv')
        all_measures.extend(measures)
        all_coef.extend([cor.statistic if not np.isnan(cor.statistic) else 0. for cor in all_corr])
        all_layers.extend([layer_combi[0] for cor in all_corr]) # layer_combi = [n], n being the number of the layer

    # make graph comparing the correlations across layers for each measure
    graph = sb.pointplot(x=all_layers, y=all_coef, hue=all_measures)
    graph.set_xticks(range(int(layers[0][0]), int(layers[-1][0])+1))
    graph.set(xlabel='Layer', ylabel='PCC')
    graph.get_figure().savefig(f'{dir_to_save}/pearson_corr_sim_{layers}_{model_name}.tiff', dpi=300)
    plt.clf()

def cross_validate(layers, paths_to_data, measures, model_name, dir_to_save, n=5, seed=1):

    # save measures, correlations, and layers
    all_measures, all_coef, all_layers = [], [], []

    for layer_combi, layer_combi_data in zip(layers, paths_to_data):

        # for specific layer, save correlations, which measures and which folds
        all_corr, all_mea, all_folds = [], [], []

        eye_move_sim_df = pd.read_csv(layer_combi_data)
        measures = [measure for measure in measures if measure in eye_move_sim_df.columns]

        for measure in measures:
            clean_df = eye_move_sim_df.dropna(subset=[measure, 'similarity'])
            # shuffle rows
            shuffled_df = clean_df.sample(frac=1, random_state=seed)
            # split rows into a number of folds for cross-validation
            df_parts = np.array_split(shuffled_df, n)
            for i, part in enumerate(df_parts):
                # for each fold, compute correlation between mean eye mov measure and mean similarity per word
                mean_measure_df = part.groupby(['trialid', 'ianum'], as_index=False)[measure].mean()
                mean_similarity_df = part.groupby(['trialid', 'ianum'], as_index=False)['similarity'].mean()
                cor = stats.pearsonr(mean_measure_df[measure].tolist(), mean_similarity_df['similarity'].tolist())
                all_corr.append(cor)
                all_folds.append(i)
                all_mea.append(measure)

        cor_df = pd.DataFrame({'measure': all_mea,
                               'fold': all_folds,
                               'coefficient': [cor.statistic if not np.isnan(cor.statistic) else 0. for cor in all_corr],
                               'p-value': [cor.pvalue for cor in all_corr]})
        cor_df.to_csv(f'{dir_to_save}/sim_{layer_combi}_{model_name}_pearsonr_corr_{n}_crossvalid.csv')
        # compute mean correlation across folds for each measure for each layer
        mean_cor_df = cor_df.groupby(['measure'], as_index=False)['coefficient'].mean()
        # print(mean_cor_df)
        # save measures for each layer
        all_measures.extend(mean_cor_df['measure'].tolist())
        # print(mean_cor_df['measure'].tolist())
        # save correlations for each layer
        all_coef.extend(mean_cor_df['coefficient'].tolist())
        # print(mean_cor_df['coefficient'].tolist())
        # save layer number
        all_layers.extend([layer_combi[0] for cor in mean_cor_df['coefficient'].tolist()]) # layer_combi = [n], n being the number of the layer
        # print([layer_combi[0] for cor in mean_cor_df['coefficient'].tolist()])

    graph = sb.pointplot(x=all_layers, y=all_coef, hue=all_measures)
    # print(all_layers)
    # print(all_coef)
    # print(all_measures)
    graph.set_xticks(range(int(layers[0][0]), int(layers[-1][0]) + 1))
    graph.set(xlabel='Layer', ylabel='PCC')
    graph.get_figure().savefig(f'{dir_to_save}/pearson_corr_sim_{layers}_{model_name}_{n}_crossvalid_mean.tiff', dpi=300)
    plt.clf()

def check_saccade_length_distribution(eye_move_df, eye_move_filepath):

    # get sac lengths
    sac_len_counts = eye_move_df['sac.out.length'].value_counts()
    # save complete distribution of saccade lengths
    sac_len_counts.to_csv(eye_move_filepath.replace('.csv', '_sac_length_distribution.csv'))
    # create column in dataframe with counts
    sac_len_counts_dict = sac_len_counts.to_dict()
    eye_move_df['sac.out.length.counts'] = eye_move_df['sac.out.length'].map(sac_len_counts_dict)
    # # find top 10 most frequent saccade lengths
    # most_freq = sac_len_counts.head(15).index.tolist()
    # print(most_freq)
    # filter dataframe with only saccade lengths within a range (-10 to +10 words)
    eye_move_df = eye_move_df[eye_move_df['sac.out.length'].isin(range(-10,11))]
    # create dist plot
    graph = sb.displot(eye_move_df, x="sac.out.length", stat='probability', discrete=True)
    plt.xticks(range(-10,11))
    filepath = eye_move_filepath.replace('.csv', '_sac_length_distribution.tiff')
    graph.savefig(filepath, dpi=300)
    plt.clf()

def check_similarity_distribution(eye_move_df, eye_move_filepath):

    graph = sb.displot(eye_move_df, x="similarity", stat='probability')
    filepath = eye_move_filepath.replace('.csv', '_sim_distribution.tiff')
    graph.savefig(filepath, dpi=300)
    plt.clf()
    eye_move_df = eye_move_df[eye_move_df['distance'].isin(range(-3,4))]
    graph = sb.displot(eye_move_df, x="similarity", hue="distance", kind="kde", palette="Set1")
    filepath = eye_move_filepath.replace('.csv', '_sim_length_distribution.tiff')
    graph.savefig(filepath, dpi=300)
    plt.clf()

def main():

    corpus_name = 'meco'
    model_name = 'gpt2'
    measures = ['dur', 'skip', 'reread']
    n = 5  # number of folds for cross-validation evaluation
    seed = 1  # seed to randomly sample trials for cross-validation
    dir_to_save = f'data/analysed/{corpus_name}/{model_name}'

    # # Graphs with pearson correlations between semantic similarity with previous context and eye mov measures
    layers = [[i] for i in range(12)]
    # paths_to_data = []  # store path to files with data from each layer (combination) to be used for eval
    # model_name = model_name.replace('/', '_')
    # for layer_combi, layer_combi_data in zip(layers, paths_to_data):
    #     eye_move_sim_filepath = f'data/processed/{corpus_name}/{model_name}/full_{model_name}_{layer_combi}_{corpus_name}_previous_context_df.csv'
    # correlate_eye2sim(layers, paths_to_data, measures, model_name, dir_to_save)
    # cross_validate(layers, paths_to_data, measures, model_name, dir_to_save, n, seed)

    # # Graph with distribution of saccade distances in fixation report
    # eye_move_filepath = f'data/processed/{corpus_name}/fixation_report_en_df.csv'
    # eye_move_df = pd.read_csv(eye_move_filepath)
    # check_saccade_length_distribution(eye_move_df, eye_move_filepath)

    # Graph with distribution of semantic similarity values
    full_data_filepath = f'data/processed/{corpus_name}/{model_name}/full_{model_name}_[11]_{corpus_name}_window_similarity_df.csv'
    full_df = pd.read_csv(full_data_filepath)
    check_similarity_distribution(full_df, full_data_filepath)

    # Check correlation between semantic similarity and positional distance
    # layers = [1,11]
    # layers_n, contexts, coeffs, pvalues = [], [], [], []
    # for layer in layers:
    #     sim_df = pd.read_csv(f'data/processed/{corpus_name}/{model_name}/similarity_window_[{layer}]_{model_name}_{corpus_name}_df.csv')
    #     # # convert distances -3 to +3 to only positive numbers (not sure what pearson does with negative numbers in the pot)
    #     sim_df['distance_transformed'] = sim_df['distance'].map({-3:1, -2:2, -1:3, 1:4, 2:5, 3:6})
    #     cor = stats.pearsonr(sim_df['similarity'].tolist(), sim_df['distance_transformed'].tolist())
    #     coeffs.append(cor.statistic)
    #     pvalues.append(cor.pvalue)
    #     print(layer, cor.statistic, cor.pvalue)
    #     # check correlation per position (left/right)
    #     sim_left = sim_df[sim_df['distance'].isin([-3,-2,-1])].copy()
    #     sim_left['distance'] = sim_left['distance'].map({-3:3,-2:2,-1:1})
    #     cor = stats.pearsonr(sim_left['similarity'].tolist(), sim_left['distance'].tolist())
    #     print('left', cor.statistic, cor.pvalue)
    #     coeffs.append(cor.statistic)
    #     pvalues.append(cor.pvalue)
    #     sim_right = sim_df[sim_df['distance'].isin([1,2,3])]
    #     cor = stats.pearsonr(sim_right['similarity'].tolist(), sim_right['distance'].tolist())
    #     print('right', cor.statistic, cor.pvalue)
    #     coeffs.append(cor.statistic)
    #     pvalues.append(cor.pvalue)
    #     layers_n.extend([layer, layer, layer])
    #     contexts.extend(['both','left','right'])
    # cor_df = pd.DataFrame({'layer': layers_n,
    #                        'context': contexts,
    #                        'coefficient': coeffs,
    #                        'p-value': pvalues})
    # cor_df.to_csv(
    #     f'data/processed/{corpus_name}/{model_name}/pearsonr_corr_sim_dist_{layers}_{model_name}_{corpus_name}.csv')

if __name__ == '__main__':
    main()
