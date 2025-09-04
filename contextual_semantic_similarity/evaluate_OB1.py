import pandas as pd
from sklearn.metrics import classification_report
from collections import defaultdict
from evaluate import average_reports, read_in_scores, test_sig_diff
import os

# Find y_true from meco data
if os.path.exists('data/processed/meco/human_fixations.csv'):
    human_fixations = pd.read_csv('data/processed/meco/human_fixations.csv')
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
    pd.DataFrame(human_fixations).to_csv('data/processed/meco/human_fixations.csv')

# Find y_true from simulation data
simulation_data_filepath = f'../data/model_output/_2025_07_04_10-49-31/meco.csv'
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

# Compute classification report for simulated fixations
splits = pd.read_csv('data/processed/meco/gpt2/optimization/cross_val_splits.txt', sep='\t')
reports = []
all_scores, all_models, all_measures = [],[],[]
# go through each val split and each text in val split
for i, val_split in enumerate(splits['test'].tolist()):
    split_y_true, split_y_pred = [],[]
    val_split = val_split.replace('[','').replace(']','').split(' ')
    for text_id in val_split:
        # filter human fixations on text id
        human_fixations_text = human_fixations[human_fixations['trialid']==int(text_id)].copy()
        # go through each human fix in text id
        for fix in human_fixations_text.itertuples():
            # find simulated fixations on the same word
            simulation_data_fix = simulation_data[(simulation_data['text_id']==int(text_id)) & (simulation_data['fixation']==fix.ianum)].copy()
            if not simulation_data_fix.empty:
                for sim_fix in simulation_data_fix.itertuples():
                    split_y_true.append(int(fix.next_saccade_distance))
                    split_y_pred.append(int(sim_fix.next_saccade_distance))
    report = pd.DataFrame(classification_report(split_y_true, split_y_pred, digits = 3, output_dict=True))
    report = report.reset_index().rename(columns={'index': 'measure'})
    reports.append(report)
    report.to_csv(f'data/processed/meco/gpt2/optimization/report_split{i}_ob1-reader.csv')
    score = report[report['measure']=='f1-score']['macro avg'].tolist()[0]
    all_scores.append(score)
    all_measures.append('f1-score')
    all_models.append('OB1-reader')
# avg_report, sd_report = average_reports(reports)
# avg_report.to_csv(f'../data/eval_output/report_ob1_avg_1.csv')
# sd_report.to_csv(f'../data/eval_output/report_ob1_sd_1.csv')

# # Compare performance OB1-reader and Classifier
# # read in classifier eval
# all_model_scores, all_models_model, all_measures_model, _ = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir='data/processed/meco/gpt2/optimization',
#                                                             measures='f1-score', models=['model'])
# assert len(all_model_scores) == len(all_scores)
# all_scores.extend(all_model_scores)
# all_models.extend(all_models_model)
# all_measures.extend(all_measures_model)
# # test significance of difference in performance between ob1-reader and classifier
# test_sig_diff(all_scores, all_models, all_measures, 'data/processed/meco/gpt2/optimization')