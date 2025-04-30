import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from collections import defaultdict

from tensorflow.python.ops.resource_variable_ops import variable_accessed


def check_saccade_length_distribution(eye_move_df, eye_move_filepath):
    # Graph with distribution of saccade distances in fixation report
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
    graph = sns.displot(eye_move_df, x="sac.out.length", stat='probability', discrete=True)
    plt.xticks(range(-10,11))
    filepath = eye_move_filepath.replace('.csv', '_sac_length_distribution.tiff')
    graph.savefig(filepath, dpi=300)
    plt.clf()

def check_similarity_distribution(eye_move_df, eye_move_filepath):
    # Graph with distribution of semantic similarity values
    graph = sns.displot(eye_move_df, x="similarity", stat='probability')
    filepath = eye_move_filepath.replace('.csv', '_sim_distribution.tiff')
    graph.savefig(filepath, dpi=300)
    plt.clf()
    eye_move_df = eye_move_df[eye_move_df['distance'].isin(range(-3,4))]
    graph = sns.displot(eye_move_df, x="similarity", hue="distance", kind="kde", palette="Set1")
    filepath = eye_move_filepath.replace('.csv', '_sim_length_distribution.tiff')
    graph.savefig(filepath, dpi=300)
    plt.clf()

def check_pred_distribution(saliency_df, variables, level, filepath):

    saliency_df_filtered = saliency_df[saliency_df['saliency_type'].isin(variables)]
    x_column = 'pred_end_relative_position'

    if 'max' in variables[0] or 'min' in variables[0]:
        if level == 'letter':
            x_column = 'pred_end_letter_relative_position'
        graph = sns.displot(saliency_df_filtered, x=x_column, hue="saliency_type",
                            stat='probability', multiple='stack', palette='deep', discrete=True)
        graph.savefig(filepath, dpi=300)

    elif 'mass' in variables[0]:
        if level == 'word':
            graph = sns.displot(saliency_df_filtered, x=x_column, hue="saliency_type",
                                stat='probability', multiple='stack', bins=[-3,-2,-1,0,1,2,3], palette="deep")
            graph.savefig(filepath, dpi=300)

    else:
        raise Exception(f'Variable types {variables} not supported')

    plt.clf()

def evaluate_saliency(df, saliency_types, output_filepath, level='word'):

    eval = defaultdict(list)
    true_col = 'end_relative_position'
    pred_col = 'pred_end_relative_position'

    if level == 'letter':
        true_col = 'end_letter_relative_position'
        pred_col = 'pred_end_letter_relative_position'

    for measure in saliency_types:

        df_measure = df[df['saliency_type'] == measure]

        for i, rows in df_measure.groupby('participant_id'):

            rows_filtered = rows[rows[true_col].notna()]
            rows_filtered = rows_filtered[rows_filtered[pred_col].notna()]
            true_y = rows_filtered[true_col].tolist()
            pred_y = rows_filtered[pred_col].tolist()

            acc, rmse = None, None
            if true_y and pred_y:
                correct = np.sum([1 if i[0] == i[1] else 0 for i in zip(true_y, pred_y)])
                acc = round(correct / len(true_y), 2)
                rmse = round(mean_squared_error(true_y, pred_y), 2)

            eval['participant_id'].append(i)
            eval['saliency_type'].append(measure)
            eval['acc'].append(acc)
            eval['rmse'].append(rmse)

    eval_df = pd.DataFrame.from_dict(eval)
    eval_df.to_csv(output_filepath, index=False)

    return eval_df

def display_eval(eval_df, variables, filepath):

    eval_df = eval_df[eval_df['saliency_type'].isin(variables)]

    # Accuracy
    if 'mass' not in variables[0]:
        graph = sns.violinplot(x=eval_df['saliency_type'], y=eval_df['acc'])
        plt.xticks(rotation=30)
        plt.tight_layout()
        graph.get_figure().savefig(f'{filepath.replace(".csv", f"_acc.tiff")}', dpi=300)
        plt.clf()

    # RMSE
    graph = sns.violinplot(x=eval_df['saliency_type'], y=eval_df['rmse'])
    plt.xticks(rotation=30)
    plt.tight_layout()
    graph.get_figure().savefig(f'{filepath.replace(".csv", f"_rmse.tiff")}', dpi=300)
    plt.clf()

def main():

    corpus_name = 'meco'
    model_name = 'gpt2'
    layers = '11'

    # # check saccade length distribution
    # eye_move_filepath = f'data/processed/{corpus_name}/fixation_report_en_df.csv'
    # eye_move_df = pd.read_csv(eye_move_filepath)
    # check_saccade_length_distribution(eye_move_df, eye_move_filepath)
    #
    # # check semantic similarity values distribution
    # full_data_filepath = f'data/processed/{corpus_name}/{model_name}/full_{model_name}_[{layers}]_{corpus_name}_window_similarity_df.csv'
    # full_df = pd.read_csv(full_data_filepath)
    # check_similarity_distribution(full_df, full_data_filepath)

    # evaluate saliency
    saliency_filepath = f'data/processed/{corpus_name}/{model_name}/saliency_{model_name}_[{layers}]_{corpus_name}.csv'
    saliency_df = pd.read_csv(saliency_filepath)
    for variable_combi in [['max_length', 'min_frequency', 'max_surprisal', 'min_semantic_similarity'],
                            ['dist_max_length', 'dist_min_frequency', 'dist_max_surprisal', 'dist_min_semantic_similarity'],
                            ['mass_length', 'mass_frequency', 'mass_surprisal', 'mass_semantic_similarity']]:
        for level in ['word', 'letter']:
            check_pred_distribution(saliency_df,
                                    variables = variable_combi,
                                    level = level,
                                    filepath = f'data/analysed/pred_distr_{model_name}_{corpus_name}_{level}_{variable_combi}.tiff')
            # add baselines
            if 'next_word' not in variable_combi and '7letter_2right' not in variable_combi:
                variable_combi.extend(['next_word','7letter_2right'])
            eval_saliency_filepath = f'data/analysed/eval_{model_name}_{corpus_name}_{level}_{variable_combi}.csv'
            eval_df = evaluate_saliency(saliency_df, variable_combi,
                                        eval_saliency_filepath, level=level)
            display_eval(eval_df, variable_combi, eval_saliency_filepath)


if __name__ == '__main__':
    main()