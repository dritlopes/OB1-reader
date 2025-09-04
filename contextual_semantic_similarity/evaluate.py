import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_rel

def average_reports(report_list):

    df_combined = pd.concat(report_list, axis=0)
    avg_report = df_combined.groupby('measure').mean(numeric_only=True)
    sd_report = df_combined.groupby('measure').std(numeric_only=True)

    return avg_report, sd_report

def evaluate_model(y_true:np.array, y_pred:np.array):

    """
    Evaluate neural network predictions. Print sklearn classification report and confusion matrix.

    :param list y_true: gold labels of test instances.
    :param list y_pred: predicted labels of test instances.
    """

    report = pd.DataFrame(classification_report(y_true, y_pred, digits = 3, output_dict=True))
    # columns = [f'true{label}' for label in np.unique(y_true)]
    # index = [f'pred{label}' for label in np.unique(y_true)]
    cfm = pd.DataFrame(confusion_matrix(y_pred,y_true))
                       # columns=columns,
                       # index=index)

    return report, cfm

def read_in_scores(splits, opt_dir, measure='f1-score', models=['model','next_word','random','ob1-reader']):

    all_values, all_models, all_positions = [], [], []

    for model in models:
        for i in splits:
            if model == 'model':  # model with all features
                filepath = f'{opt_dir}/report_split{i}.csv'
            elif model in ['random', 'next_word']:  # baselines
                filepath = f'{opt_dir}/report_split{i}_baseline_{model}.csv'
            else:  # if feature combi for feature ablation; or ob1-reader
                filepath = f'{opt_dir}/report_split{i}_{model}.csv'
            df = pd.read_csv(filepath)
            for position in ['-3', '-2', '-1', '0', '1', '2', '3']:
                if measure == 'accuracy':
                    value = df['accuracy'].tolist()[0]
                else:
                    # value = df[df['measure'] == measure]['macro avg'].tolist()[0]
                    value = df[df['measure'] == measure][position].tolist()[0]
                all_values.append(value)
                all_models.append(model)
                all_positions.append(position)

    return all_values, all_models, all_positions

def test_sig_diff(all_values, all_models, all_measures, opt_dir):

    df = pd.DataFrame({'score': all_values, 'model': all_models, 'measure': all_measures})

    # for each measure, e.g. acc and f1-score, take scores of each model, combine them in pairs, and perform t-test
    for measure, rows in df.groupby('measure'):
        score_dict = defaultdict(list)
        for model, scores in rows.groupby('model'):
            score_dict[model] = scores['score'].tolist()
        for model_combi in combinations(rows['model'].unique().tolist(), 2):
            result = ttest_rel(score_dict[model_combi[0]], score_dict[model_combi[1]])
            with open(f'{opt_dir}/t-test_{measure}_{model_combi}.csv', 'w') as f:
                f.write('t-statistic\tp-value\tdf\n')
                f.write(f'{result.statistic}\t{result.pvalue}\t{result.df}\n')

def display_prediction_distribution(true, pred, filepath, title='', col=None):

    true = np.array(true)
    pred = np.array(pred)
    # create x, hue and col
    targets = np.concatenate((true, pred), axis=0)
    target_types = [0 for i in true] + [1 for i in pred] # if the value is true or pred
    if col:
        col = np.array(col)
        col = np.concatenate((col, col), axis=0)
    # display pred distr and true dist
    graph = sns.displot(x=targets, hue=target_types, col=col, stat='probability', multiple='dodge', palette='deep', bins=[-3,-2,-1,0,1,2,3,4], legend=False)
    plt.legend(title=title, labels=['predicted', 'true'])
    plt.xticks([-3, -2, -1, 0, 1, 2, 3])  # remove bin edge 4
    graph.savefig(filepath, dpi=300)
    plt.clf()

def display_error(errors, conditions, loss_function, filepath, hue=None, hue_name=None, col=None, col_name=None):

    data = pd.DataFrame({'condition': conditions,
                        'error': errors})
    if col:
        if col_name:
            data[col_name] = col
        else:
            raise ValueError('If col is given, a col_name must be provided.')
    if hue:
        if hue_name:
            data[hue_name] = hue
        else:
            raise ValueError('If hue is given, a hue_name must be provided.')

    graph = sns.catplot(data=data,
                        x='condition', y='error', hue=hue_name, col=col_name,
                        kind="bar", palette='deep', col_wrap=3)
    graph.set(ylabel=loss_function)
    graph.savefig(filepath, dpi=300)
    plt.clf()

def display_eval(all_values, all_models, all_positions, measure, filepath, col=None, col_name=None):

    data = pd.DataFrame({'position': all_positions,
                         'score': all_values,
                         'condition': all_models})

    if col:
        if col_name:
            data[col_name] = col
        else:
            raise ValueError('If col is given, a col_name must be provided.')

    graph = sns.catplot(data= data,
                        x='position', y='score', hue='condition',
                        kind='bar', palette='deep', col=col_name)

    graph.set(xlabel='word position', ylabel=measure)
    graph.savefig(filepath, dpi=300)
    plt.clf()

def display_eval_features(all_values, all_models, all_positions, measure, filepath):

    data = pd.DataFrame({'position': all_positions,
                         'score': all_values,
                         'condition': all_models})

    condition_map = {'length,surprisal,frequency,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration': 'full model',
                     'frequency,surprisal,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration': 'length',
                     'length,surprisal,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration': 'frequency',
                     'length,frequency,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration': 'surprisal',
                     'length,frequency,surprisal,embedding,previous_sacc_distance,previous_fix_duration': 'has_been_fixed',
                     'length,frequency,surprisal,has_been_fixated,previous_sacc_distance,previous_fix_duration': 'embedding',
                     'length,frequency,surprisal,has_been_fixated,embedding,previous_fix_duration': 'previous_sacc_distance',
                     'length,frequency,surprisal,has_been_fixated,embedding,previous_sacc_distance': 'previous_fix_duration'}
    conditions_rename = []
    for condition in data['condition'].tolist():
        conditions_rename.append(condition_map[condition])
    data['condition'] = conditions_rename

    data = data[data['position'].isin(['-1','0','1','2','3'])]

    # diff of split avg from full model
    avg_data = data.groupby(by=['condition','position'], as_index=False)['score'].mean()
    diff_score = []
    for score, model, position in zip(avg_data['score'].tolist(), avg_data['condition'].tolist(), avg_data['position'].tolist()):
        full_model_score = avg_data[(avg_data['condition']=='full model') & (avg_data['position']==position)]['score'].tolist()[0]
        diff = full_model_score - score
        diff_score.append(diff)
    avg_data['diff_score'] = diff_score

    avg_data.rename(columns={'diff_score': f'difference in {measure}'}, inplace=True)

    for position, rows in avg_data.groupby('position'):
        graph = sns.catplot(data= rows,
                            x='difference in f1-score', y='condition',
                            kind='bar', palette='deep')
        plt.axvline(x=0, color='black', linestyle='--')
        graph.fig.suptitle(f'{position}')
        graph.set( ylabel='ablated feature')
        plt.tight_layout()
        graph.savefig(filepath.replace('.tiff',f'{position}.tiff'), dpi=300)
        plt.clf()

def main():

    # # display eval models
    opt_dir = 'data/processed/meco/gpt2/optimization'
    # baselines = 'ob1-reader,random,next_word'
    measure = 'f1-score'
    # models = ['model'] + baselines.split(',')
    # all_values, all_models, all_positions = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir=opt_dir, measure=measure,
    #                                                       models=models)
    # display_eval(all_values, all_models, all_positions,  measure, filepath=f'{opt_dir}/eval_all_models.tiff')
    # # test_sig_diff(all_values, all_models, all_positions, opt_dir)

    # display eval feature ablation
    features_to_select = ['length,surprisal,frequency,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration', # full
                           'frequency,surprisal,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration',
                            'length,surprisal,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration',
                            'length,frequency,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration',
                            'length,frequency,surprisal,embedding,previous_sacc_distance,previous_fix_duration',
                            'length,frequency,surprisal,has_been_fixated,previous_sacc_distance,previous_fix_duration',
                            'length,frequency,surprisal,has_been_fixated,embedding,previous_fix_duration',
                            'length,frequency,surprisal,has_been_fixated,embedding,previous_sacc_distance']
    ablation_type = 'mean'
    all_values, all_err_models, all_positions = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir=opt_dir,
                                                            measure=measure,
                                                            models=features_to_select)
    display_eval_features(all_values, all_err_models, all_positions, measure,
                 filepath=f'{opt_dir}/eval_feature_ablation_{ablation_type}_position.tiff')
    # test_sig_diff(all_values, all_err_models, all_positions, opt_dir)

if __name__ == '__main__':
    main()