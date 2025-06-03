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

def display_eval(all_values, all_models, all_measures, filepath):

    graph = sns.catplot(data= pd.DataFrame({'measure': all_measures,
                                            'score': all_values,
                                            'condition': all_models}),
                        x='measure', y='score', hue='condition',
                        kind='bar', palette='deep')

    # graph.set(xlabel='measure')
    graph.savefig(filepath, dpi=300)
    plt.clf()
