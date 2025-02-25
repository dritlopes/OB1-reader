from scipy import stats
import pandas as pd
import seaborn as sb
import numpy as np
import os

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

def evaluation(layers, paths_to_data, measures, model_name, n, seed, dir_to_save):

    if not os.path.isdir(dir_to_save):
        os.mkdir(dir_to_save)

    correlate_eye2sim(layers, paths_to_data, measures, model_name, dir_to_save)
    cross_validate(layers, paths_to_data, measures, model_name, dir_to_save, n, seed)
