import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from train_classifier import average_reports
import pickle
import os

# Find y_true from meco data
if os.path.exists('data/processed/human_fixations.pkl'):
    with open('data/processed/human_fixations.pkl', 'rb') as f:
        human_fixations = pickle.load(f)
else:
    eye_data_filepath = f'../contextual_semantic_similarity/data/processed/meco/gpt2/full_gpt2_[1]_meco_window_cleaned.csv'
    eye_data = pd.read_csv(eye_data_filepath)
    human_fixations = defaultdict(list)
    for i, context in eye_data.groupby(['participant_id', 'trialid', 'fixid']):
        human_fixations['participant_id'].append(i[0])
        human_fixations['trialid'].append(i[1])
        human_fixations['fixid'].append(i[2])
        human_fixations['ianum'].append(context['ianum'].tolist()[0])
        human_fixations['ia'].append(context['ia'].tolist()[0])
        for context_word in context.itertuples():
            if context_word.landing_target:
                human_fixations['next_saccade_distance'].append(context_word.distance)
    with open('data/processed/human_fixations.pkl', 'wb') as f:
        pickle.dump(human_fixations, f)

# Find y_true from simulation data
simulation_data_filepath = f'../data/model_output/_2025_07_03_15-16-42/meco.csv'
simulation_data = pd.read_csv(simulation_data_filepath, sep='\t')
# find next saccade target
if 'next_saccade_distance' not in simulation_data.columns:
    distances = []
    for id, fixations in simulation_data.groupby(['simulation_id','text_id']):
        ianums = fixations['fixation'].tolist()
        for i, ianum in enumerate(ianums):
            # if not last fixation, register the number of words between this and the next fixation
            if i + 1 < len(ianums):
                distances.append(ianums[i+1] - ianum)
            # if last fixation, no sacc.out distance
            else:
                distances.append(None)
    simulation_data['next_saccade_distance'] = distances
    simulation_data.to_csv(simulation_data_filepath, sep='\t')

# Compute classification report for these fixations
reports, confusion_matrices = [],[]
for simulation_id, simulation in simulation_data.groupby('simulation_id'):
    y_true, y_pred = [],[]
    # filter words that were fixated by both human and model
    for participant_id, text_id, fixated_word, word, target in zip(human_fixations['participant_id'],
                                                              human_fixations['trialid'],
                                                              human_fixations['ianum'],
                                                              human_fixations['ia'],
                                                              human_fixations['next_saccade_distance']):

        simulation_fixated_words = [f'{textid}-{fixated_ianum}' for textid, fixated_ianum in zip(simulation['text_id'],
                                                                                             simulation['fixation'])]
        if f'{text_id}-{fixated_word}' in simulation_fixated_words:
            sim_fixations = simulation[(simulation['text_id']==text_id) & (simulation['fixation'] == fixated_word)]
            for sim_fix in sim_fixations.itertuples():
                y_true.append(int(target))
                y_pred.append(int(sim_fix.next_saccade_distance))
    report = pd.DataFrame(classification_report(y_true, y_pred, digits = 3, output_dict=True))
    report = report.reset_index().rename(columns={'index': 'measure'})
    reports.append(report)
avg_report, sd_report = average_reports(reports)
avg_report.to_csv(f'../data/eval_output/report_ob1_avg.csv')
sd_report.to_csv(f'../data/eval_output/report_ob1_sd.csv')
