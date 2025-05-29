import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    graph = sns.displot(x=targets, hue=target_types, col=col, col_wrap=3, stat='probability', multiple='dodge', palette='deep', bins=[-3,-2,-1,0,1,2,3,4], legend=False)
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

def display_eval(splits, opt_dir, measures='f1-score,accuracy', models='model,next_word,7letter_2right,random', feature_combi=''):

    all_values, all_models, all_measures = [], [], []

    for model in models.split(','):
        for i in splits:
            if model == 'model' and not feature_combi: # model with all features
                filepath = f'{opt_dir}/report_split{i}.csv'
            elif not feature_combi: # baselines
                filepath = f'{opt_dir}/report_split{i}_baseline_{model}.csv'
            else: # if feature combi for feature ablation
                filepath = f'{opt_dir}/report_split{i}_{feature_combi}.csv'
            df = pd.read_csv(filepath)
            for measure in measures.split(','):
                if measure == 'accuracy':
                    value = df['accuracy'].tolist()[0]
                else:
                    value = df[df['Unnamed: 0']==measure]['macro avg'].tolist()[0]
                all_values.append(value)
                all_models.append(model)
                all_measures.append(measure)

    filepath = f'{opt_dir}/eval.tiff'
    if feature_combi: filepath = f'{opt_dir}/eval_{feature_combi}.tiff'
    graph = sns.catplot(data= pd.DataFrame({'measure': all_measures,
                                            'score': all_values,
                                            'condition': all_models}),
                        x='measure', y='score', hue='condition',
                        kind='bar', palette='deep')
    # graph.set(xlabel='measure')
    graph.savefig(filepath, dpi=300)
    plt.clf()
