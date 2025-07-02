from collections import defaultdict

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import ttest_rel
import os
from prepare_data import convert_data_to_tensors, split_data, FixationDataset, load_baseline_tensors, clean_tensors
from visualizations import display_eval, display_prediction_distribution
import pickle

# nn class definition
class Classifier(nn.Module):

    # initialise the neural network
    def __init__(self,
                 input_nodes,
                 hidden_nodes,
                 output_nodes,
                 learning_rate=0.001,
                 drop_out=0.1):
        """
        Define the neural network by setting the number of input nodes, hidden nodes and output nodes, as well as the learning rate.
        Initialise weight matrices.
        Define activation function.

        :param int input_nodes: number of input nodes in the input layer, corresponding to the length of the input vector
        :param int hidden_nodes: number of hidden nodes in a hidden layer
        :param int output_nodes: number of output nodes in the output layer, corresponding to the number of classes
        :param float learning_rate: the learning rate at which the connection weights are updated
        """

        super().__init__()

        self.lr = learning_rate
        self.drop_out = drop_out
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # define model to process low-level linguistic features
        # self.textual_input_dim = lstm_input_nodes # number of features for each word in sequence
        # self.lstm_units = lstm_hidden_nodes # number  of hidden states
        # self.lstm_layer = nn.LSTM(self.textual_input_dim, self.lstm_units, batch_first=True)
        # self.convs = nn.ModuleList([
        #     nn.Conv1d(in_channels=4, # one filter per feature
        #               out_channels=10, # how many filters per kernel size
        #               kernel_size=k) # sliding window, the number of consecutive words the filter spans
        #     for k in [2,3,4]
        # ])
        # self.fc = nn.Linear(len([2,3,4]) * 10, 7)

        # # define model to process previous fixation features
        # self.fixation_input_dim = 2
        # self.hidden_nodes = 2
        # self.nn_layer = nn.Sequential(nn.Linear(self.fixation_input_dim, self.hidden_nodes),
        #                                  nn.ReLU())

        # define model to process previous context features
        # self.lstm_layer = nn.LSTM(input_size= 772,
        #                           hidden_size= 32,
        #                           batch_first=True)

        # # define model to process upcoming context features
        # self.convs = nn.ModuleList([
        #     nn.Conv1d(in_channels=4, # one filter per feature
        #               out_channels=10, # how many filters per kernel size
        #               kernel_size=k) # sliding window, the number of consecutive words the filter spans
        #     for k in [2,3]])

        # define model to process embedding(s)
        # self.lstm_layer = nn.LSTM(input_size= 768,
        #                           hidden_size= 128,
        #                           batch_first=True)

        # define model to process fix features
        # self.fc_layer = nn.Sequential(nn.Linear(2, 4),
        #                               nn.ReLU())

        # define classifier layers
        # self.input_nodes = self.lstm_units
        # self.hidden_nodes = self.lstm_units
        # self.input_nodes = (773 * 4) + (5 * 3)
        # self.hidden_nodes = 256
        # self.output_nodes = 7
        self.classifier_layer_stack = nn.Sequential(nn.Linear(self.input_nodes, self.hidden_nodes),
                                         nn.ReLU(),
                                         nn.Dropout(self.drop_out),
                                         nn.Linear(self.hidden_nodes, self.output_nodes))

    def forward(self, inputs): # word_inputs, previous_fix_inputs, llm_inputs # previous_context_input, upcoming_context_input

        # shape of lsmt_output = (batch size, sequence length, features)
        # shape of hidden output = (number of directions, batch size, hidden states)
        # lstm_out, (hidden, cell) = self.lstm_layer(previous_context_input)
        # pooled_lsmt_out = torch.mean(lstm_out, dim=1)
        # logits = self.classifier_layer_stack(pooled_lsmt_out)
        # hidden_output = hidden.squeeze(0)
        # logits = self.classifier_layer_stack(hidden_output)

        # x = upcoming_context_input.permute(0, 2, 1)  # (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim, seq_len) for Conv1d
        # x = [nn.functional.relu(conv(x)) for conv in self.convs]  # list of (batch, num_filters, ~)
        # x = [nn.functional.max_pool1d(i, i.shape[2]).squeeze(2) for i in x]  # (batch, num_filters)
        # cnn_out = torch.cat(x, dim=1) # (batch, num_filters * len(kernel_sizes))
        # logits = self.fc(x)

        # x = word_inputs.permute(0, 2, 1)  # (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim, seq_len) for Conv1d
        # x = [nn.functional.relu(conv(x)) for conv in self.convs]  # list of (batch, num_filters, ~)
        # x = [nn.functional.max_pool1d(i, i.shape[2]).squeeze(2) for i in x]  # (batch, num_filters)
        # cnn_out = torch.cat(x, dim=1) # (batch, num_filters * len(kernel_sizes))
        #
        # lstm_out, (hidden, cell) = self.lstm_layer(llm_inputs)
        #
        # fc_out = self.fc_layer(fix_inputs)

        # combined_input = torch.cat((pooled_lsmt_out, cnn_out), dim=1)
        # combined_input = torch.cat((cnn_out, fc_out, lstm_out), dim=-1)

        logits = self.classifier_layer_stack(inputs)

        return logits

def train_model(model:nn.Module,
                training_dataset:DataLoader,
                val_dataset:DataLoader,
                epochs:int,
                device,
                loss_weights:torch.Tensor|None=None,
                display:bool=False,
                display_dir:str='',
                n_split:int=0):

    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    all_loss, all_acc, split_type, i_epoch = [],[],[],[]

    start_time = time.perf_counter()

    for epoch in range(epochs):

        epoch_start_time = time.perf_counter()

        epoch_loss, epoch_acc = 0, []
        epoch_val_loss, epoch_val_acc = 0, []

        # Put model in train mode (e.g. update weights with forward pass)
        # https://docs.pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train
        model.train()

        # for batch_x_word, batch_x_fix, batch_x_embed, batch_y in training_dataset:
        for batch_x, batch_y in training_dataset:
        # for batch_x_previous, batch_x_upcoming, batch_y in training_dataset:
        # for batch_x_word, batch_x_embed, batch_y in training_dataset:

            # batch_x_word, batch_x_fix, batch_x_embed, batch_y = (
            #     batch_x_word.to(device), batch_x_fix.to(device), batch_x_embed.to(device), batch_y.to(device))
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # batch_x_previous, batch_x_upcoming, batch_y = batch_x_previous.to(device), batch_x_upcoming.to(device), batch_y.to(device)
            # batch_x_word, batch_x_embed, batch_y = batch_x_word.to(device), batch_x_embed.to(device), batch_y.to(device)

            # Forward pass (model outputs raw logits)
            y_logits = model(batch_x)
            # y_logits = model(batch_x_word, batch_x_fix, batch_x_embed)
            # y_logits = model(batch_x_previous, batch_x_upcoming)
            # y_logits = model(batch_x_word, batch_x_embed)

            # Calculate loss
            loss = cross_entropy(y_logits,
                                 batch_y,
                                 weight=loss_weights)
            epoch_loss += loss.item()

            # Optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy for inspection
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            acc = torch.eq(batch_y, y_pred).sum() / y_pred.shape[0] * 100
            epoch_acc.append(acc.item())

        all_loss.append(round(epoch_loss,2))
        all_acc.append(round(np.mean(epoch_acc),2))
        split_type.append('train')
        i_epoch.append(epoch)

        # Validation
        model.eval()

        # for val_batch_x_word, val_batch_x_fix, val_batch_x_embed, val_batch_y in val_dataset:
        for val_batch_x, val_batch_y in val_dataset:
        # for val_batch_x_previous, val_batch_x_upcoming, val_batch_y in val_dataset:
        # for val_batch_x_word, val_batch_x_embed, val_batch_y in val_dataset:

            with torch.inference_mode():

                # val_batch_x_word, val_batch_x_fix, val_batch_x_embed, val_batch_y = val_batch_x_word.to(device), val_batch_x_fix.to(device), val_batch_x_embed.to(device), val_batch_y.to(device)
                val_batch_x, val_batch_y = val_batch_x.to(device), val_batch_y.to(device)
                # val_batch_x_previous, val_batch_x_upcoming, val_batch_y = val_batch_x_previous.to(device), val_batch_x_upcoming.to(device), val_batch_y.to(device)
                # val_batch_x_word, val_batch_x_embed, val_batch_y = val_batch_x_word.to(device), val_batch_x_embed.to(device), val_batch_y.to(device)
                # val_logits = model(val_batch_x_word, val_batch_x_fix, val_batch_x_embed)
                val_logits = model(val_batch_x)
                # val_logits = model(val_batch_x_previous, val_batch_x_upcoming)
                # val_logits = model(val_batch_x_word, val_batch_x_embed)
                val_loss = cross_entropy(val_logits,
                                          val_batch_y,
                                          weight=loss_weights)
                epoch_val_loss += val_loss.item()
                val_pred = torch.argmax(torch.softmax(val_logits,dim=1), dim=1)
                val_acc = torch.eq(val_batch_y, val_pred).sum() / val_batch_y.shape[0] * 100
                epoch_val_acc.append(val_acc.item())

        all_loss.append(round(epoch_val_loss,2))
        all_acc.append(round(np.mean(epoch_val_acc),2))
        split_type.append('val')
        i_epoch.append(epoch)

        # Print out loss and acc every epoch
        if epoch % 1 == 0:
            print(
                f"Epoch: {epoch} | Loss: {epoch_loss:.5f}, Accuracy: {np.mean(epoch_acc):.2f}% | Val loss: {epoch_val_loss:.5f}, Val acc: {np.mean(epoch_val_acc):.2f}%")
            print(f"Epoch time: {(time.perf_counter() - epoch_start_time)/60} minutes")

    print(f"Total training time: {(time.perf_counter() - start_time)/60} minutes: ")

    if display:
        graph = sns.relplot(x=i_epoch, y=all_loss, hue=split_type, kind='line')
        graph.set(ylabel='loss', xlabel='epoch', title='Training loss')
        if display_dir:
            graph.savefig(f'{display_dir}/training_loss_split{n_split}.tiff', dpi=300)
        plt.clf()
        graph = sns.relplot(x=i_epoch, y=all_acc, hue=split_type, kind='line')
        graph.set(ylabel='accuracy', xlabel='epoch', title='Training accuracy')
        if display_dir:
            graph.savefig(f'{display_dir}/training_acc_split{n_split}.tiff', dpi=300)
        plt.clf()

def test_model(model, test_dataset, device):

    test_predictions, test_labels = [], []

    model.eval()

    with torch.inference_mode():

        # for test_x_word, test_x_fix, test_x_embed, test_y in test_dataset:
        for test_x, test_y in test_dataset:
        # for test_x_previous, test_x_upcoming, test_y in test_dataset:
        # for test_x_word, test_x_embed, test_y in test_dataset:

            # test_x_word, test_x_fix, test_x_embed, test_y = test_x_word.to(device), test_x_fix.to(device), test_x_embed.to(device), test_y.to(device)
            # test_logits = model(test_x_word, test_x_fix, test_x_embed)
            test_x, test_y = test_x.to(device), test_y.to(device)
            test_logits = model(test_x)
            # test_x_previous, test_x_upcoming, test_y = test_x_previous.to(device), test_x_upcoming.to(device), test_y.to(device)
            # test_logits = model(test_x_previous, test_x_upcoming)
            # test_x_word, test_x_embed, test_y = test_x_word.to(device), test_x_embed.to(device), test_y.to(device)
            # test_logits = model(test_x_word, test_x_embed)
            test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)
            test_predictions.append(test_pred)
            test_labels.append(test_y)

    test_pred = torch.cat(test_predictions, dim=0)
    test_true = torch.cat(test_labels, dim=0)

    assert test_pred.shape == test_true.shape

    test_pred = test_pred.detach().cpu().numpy()
    test_true = test_true.detach().cpu().numpy()

    return test_pred, test_true

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

def average_reports(report_list):


    df_combined = pd.concat(report_list, axis=0)
    avg_report = df_combined.groupby('measure').mean(numeric_only=True)
    sd_report = df_combined.groupby('measure').std(numeric_only=True)

    return avg_report, sd_report

def read_in_scores(splits, opt_dir, measures='f1-score,accuracy',
                   models=['model','next_word','7letter_2right','random'],participant_ids=''):

    all_values, all_models, all_measures, all_participants = [], [], [], []

    # if participant_ids:
    #     for participant in participant_ids.split(','):
    #         for model in models:
    #             for i in splits:
    #                 if model == 'model' and not feature_combi:  # model with all features
    #                     filepath = f'{opt_dir}/report_split{i}_participant{participant}.csv'
    #                 elif not feature_combi:  # baselines
    #                     filepath = f'{opt_dir}/report_split{i}_baseline_{model}_participant{participant}.csv'
    #                 else:  # if feature combi for feature ablation
    #                     filepath = f'{opt_dir}/report_split{i}_{feature_combi}_participant{participant}.csv'
    #                 df = pd.read_csv(filepath)
    #                 for measure in measures.split(','):
    #                     if measure == 'accuracy':
    #                         value = df['accuracy'].tolist()[0]
    #                     else:
    #                         value = df[df['Unnamed: 0'] == measure]['macro avg'].tolist()[0]
    #                     all_values.append(value)
    #                     all_models.append(model)
    #                     all_measures.append(measure)
    #                     all_participants.append(participant)
    # else:
    for model in models:
        for i in splits:
            if model == 'model':  # model with all features
                filepath = f'{opt_dir}/report_split{i}.csv'
            elif model in ['random', 'next_word']:  # baselines
                filepath = f'{opt_dir}/report_split{i}_baseline_{model}.csv'
            else:  # if feature combi for feature ablation
                filepath = f'{opt_dir}/report_split{i}_{model}.csv'
            df = pd.read_csv(filepath)
            for measure in measures.split(','):
                if measure == 'accuracy':
                    value = df['accuracy'].tolist()[0]
                else:
                    value = df[df['measure'] == measure]['macro avg'].tolist()[0]
                all_values.append(value)
                all_models.append(model)
                all_measures.append(measure)

    return all_values, all_models, all_measures, all_participants

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

def train_all(eye_data:pd.DataFrame,
              split_indices:dict[str, list[str]],
              opt_dir:str,
              tensor_dir:str,
              params_dataloader, params_classifier, device, epochs, baselines):

    all_targets, all_predictions, all_models = [], [], []
    reports, random_reports, majority_reports = [], [], []

    for i, split in enumerate(split_indices):

        print(f'Split: {i}')

        # train using all participant data for the texts in training split (split based on text)
        train_split_ids = [f'{participant_id},{text_id}' for text_id in split['train_index']
                           for participant_id in eye_data[eye_data['trialid']==text_id]['participant_id'].unique()]
        training_set = FixationDataset(train_split_ids, tensor_dir)
        training_generator = DataLoader(training_set, **params_dataloader)
        # validate using all participant data for the texts in val split (split based on text)
        val_split_ids = [f'{participant_id},{text_id}' for text_id in split['test_index']
                           for participant_id in eye_data[eye_data['trialid']==text_id]['participant_id'].unique()]
        val_set = FixationDataset(val_split_ids, tensor_dir)
        val_generator = DataLoader(val_set, **params_dataloader)

        # compute weights for the classes based on class distribution (more weight for infrequent classes)
        # loss_weights = compute_class_weight("balanced",
        #                                     classes=np.unique(training_set.y_tensor),
        #                                     y=np.array(training_set.y_tensor))
        # loss_weights = torch.FloatTensor(loss_weights).to(device)

        params_classifier['input_nodes'] = (training_set.x_word_tensor.shape[-1] +
                                            training_set.x_embed_tensor.shape[-1] +
                                            training_set.x_fix_tensor.shape[-1])

        model = Classifier(**params_classifier).to(device)
        train_model(model=model,
                    training_dataset=training_generator,
                    val_dataset=val_generator,
                    epochs=epochs,
                    device=device,
                    display=True,
                    display_dir=opt_dir,
                    n_split=i,
                    loss_weights=None)
        torch.save(model, f'{opt_dir}/classifier_split{i}.pth')
        # model = torch.load(f'{opt_dir}/classifier_split{i}_loss_weights.pth', map_location=device)

        # Evaluate model with validation set
        print('Testing training model...')
        y_pred, y_true = test_model(model=model, test_dataset=val_generator, device=device)
        # convert labels back to -3 to +3
        y_true = y_true - 3
        y_pred = y_pred - 3
        report, cf_matrix = evaluate_model(y_true, y_pred)
        report = report.reset_index().rename(columns={'index': 'measure'})
        # report = pd.read_csv(f'{opt_dir}/report_split{i}.csv')
        reports.append(report)
        report.to_csv(f'{opt_dir}/report_split{i}.csv')
        cf_matrix.to_csv(f'{opt_dir}/confusion_matrix_split{i}.csv')
        all_targets.append(y_true)
        all_predictions.append(y_pred)
        all_models.extend(['model' for i in range(y_true.shape[0])])

        # evaluate baselines
        print('Computing and evaluating baselines...')
        for baseline in baselines.split(','):  # '7letter_2right'
            if baseline in ['next_word', '7letter_2right']:
                y_pred = load_baseline_tensors(val_split_ids, baseline, tensor_dir)
                y_true = val_set.y_tensor
                y_true, y_pred = clean_tensors(y_true, y_pred)
                y_true = y_true - torch.tensor(3)
                y_pred = y_pred.detach().cpu().numpy()
                y_true = y_true.detach().cpu().numpy()
            else:
                print('Training random model...')
                model = Classifier(**params_classifier).to(device)
                training_set = FixationDataset(train_split_ids, tensor_dir, random=True)
                training_generator = DataLoader(training_set, **params_dataloader)
                val_set = FixationDataset(val_split_ids, tensor_dir, random=True)
                val_generator = DataLoader(val_set, **params_dataloader)
                train_model(model=model,
                            training_dataset=training_generator,
                            val_dataset=val_generator,
                            epochs=epochs,
                            device=device,
                            n_split=i,
                            loss_weights=None)
                torch.save(model, f'{opt_dir}/classifier_split{i}_random.pth')
                # model = torch.load(f'{opt_dir}/classifier_split{i}_random.pth', map_location=device)
                y_pred, y_true = test_model(model=model, test_dataset=val_generator, device=device)
                # convert labels back to -3 to +3
                y_true = y_true - 3
                y_pred = y_pred - 3
            report, cf_matrix = evaluate_model(y_true, y_pred)
            report = report.reset_index().rename(columns={'index': 'measure'})
            report.to_csv(f'{opt_dir}/report_split{i}_baseline_{baseline}.csv')
            # report = pd.read_csv(f'{opt_dir}/report_split{i}_baseline_{baseline}.csv')
            if baseline == 'random':
                random_reports.append(report)
            elif baseline == 'next_word':
                majority_reports.append(report)
            cf_matrix.to_csv(f'{opt_dir}/confusion_matrix_split{i}_baseline_{baseline}.csv')
            all_targets.append(y_true)
            all_predictions.append(y_pred)
            all_models.extend([baseline for i in range(y_true.shape[0])])

    avg_report, sd_report = average_reports(reports)
    avg_report.to_csv(f'{opt_dir}/report_avg.csv')
    sd_report.to_csv(f'{opt_dir}/report_sd.csv')
    avg_report_random, sd_report_random = average_reports(random_reports)
    avg_report_random.to_csv(f'{opt_dir}/report_avg_random.csv')
    sd_report_random.to_csv(f'{opt_dir}/report_sd_random.csv')
    avg_report_majority, sd_report_majority = average_reports(majority_reports)
    avg_report_majority.to_csv(f'{opt_dir}/report_avg_majority.csv')
    sd_report_majority.to_csv(f'{opt_dir}/report_sd_majority.csv')

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    print('Creating graphs of validation results...')
    models = ['model'] + baselines.split(',')
    display_prediction_distribution(all_targets, all_predictions,
                                    filepath=f'{opt_dir}/distribution_nn_all_val_splits.tiff', col=all_models)

    all_values, all_models, all_measures, _ = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir=opt_dir,
                                                             measures='f1-score,accuracy', models=models)
    display_eval(all_values, all_models, all_measures, filepath=f'{opt_dir}/eval.tiff')
    test_sig_diff(all_values, all_models, all_measures, opt_dir)

def feature_ablation(eye_data, params_classifier, split_indices, device, opt_dir, tensor_dir, params_dataloader, epochs,
                     baselines, feature_map, features_to_select, ablation_type):

    print('Feature ablation...')
    all_targets, all_predictions, all_models = [], [], []

    for feature_combi in features_to_select:

        reports = []

        print(f'features: {feature_combi}')

        for i, split in enumerate(split_indices):

            train_split_ids = [f'{participant_id},{text_id}' for text_id in split['train_index']
                               for participant_id in
                               eye_data[eye_data['trialid'] == text_id]['participant_id'].unique()]
            training_set = FixationDataset(train_split_ids, tensor_dir, features_to_select=feature_combi, feature_map=feature_map, ablation_type=ablation_type)
            training_generator = DataLoader(training_set, **params_dataloader)
            val_split_ids = [f'{participant_id},{text_id}' for text_id in split['test_index']
                             for participant_id in eye_data[eye_data['trialid'] == text_id]['participant_id'].unique()]
            val_set = FixationDataset(val_split_ids, tensor_dir, features_to_select=feature_combi, feature_map=feature_map, ablation_type=ablation_type)
            val_generator = DataLoader(val_set, **params_dataloader)

            # loss_weights = compute_class_weight("balanced",
            #                                     classes=np.unique(training_set.y_tensor),
            #                                     y=np.array(training_set.y_tensor))
            # loss_weights = torch.FloatTensor(loss_weights).to(device)

            params_classifier['input_nodes'] = 0
            if not torch.all(training_set.x_word_tensor==0):
                params_classifier['input_nodes'] += training_set.x_word_tensor.shape[-1]
            if not torch.all(training_set.x_fix_tensor==0):
                params_classifier['input_nodes'] += training_set.x_fix_tensor.shape[-1]
            if not torch.all(training_set.x_embed_tensor==0):
                params_classifier['input_nodes'] += training_set.x_embed_tensor.shape[-1]

            # decrease number of hidden nodes if no embedding in input
            if torch.all(training_set.x_embed_tensor==0):
                params_classifier['hidden_nodes'] = params_classifier['input_nodes']

            model = Classifier(**params_classifier).to(device)
            train_model(model=model,
                        training_dataset=training_generator,
                        val_dataset=val_generator,
                        epochs=epochs,
                        device=device,
                        n_split=i,
                        loss_weights=None)

            torch.save(model, f'{opt_dir}/classifier_split{i}_{ablation_type}_{feature_combi}.pth')
            # model = torch.load(f'{opt_dir}/classifier_split{i}_{feature_combi}.pth', map_location=device)

            y_pred, y_true = test_model(model=model, test_dataset=val_generator, device=device)
            y_true = y_true - 3
            y_pred = y_pred - 3
            report, cf_matrix = evaluate_model(y_true, y_pred)
            report = report.reset_index().rename(columns={'index': 'measure'})
            reports.append(report)
            report.to_csv(f'{opt_dir}/report_split{i}_{feature_combi}.csv')
            cf_matrix.to_csv(f'{opt_dir}/confusion_matrix_split{i}_{ablation_type}_{feature_combi}.csv')
            all_targets.append(y_true)
            all_predictions.append(y_pred)
            all_models.extend([feature_combi for i in range(y_true.shape[0])])

        avg_report, sd_report = average_reports(reports)
        avg_report.to_csv(f'{opt_dir}/report_avg_{feature_combi}.csv')
        sd_report.to_csv(f'{opt_dir}/report_sd_{feature_combi}.csv')

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    display_prediction_distribution(all_targets, all_predictions,
                                    filepath=f'{opt_dir}/distribution_nn_feature_ablation_{ablation_type}_all_val_splits.tiff',
                                    col=all_models)
    features_to_select.extend(baselines.split(','))
    all_values, all_err_models, all_measures, _ = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir=opt_dir,
                                                                 measures='f1-score,accuracy',
                                                                 models= features_to_select)
    display_eval(all_values, all_err_models, all_measures, filepath=f'{opt_dir}/eval_feature_ablation_{ablation_type}.tiff')
    test_sig_diff(all_values, all_err_models, all_measures, opt_dir)


def main():

    eye_data_filepath = f'data/processed/meco/gpt2/full_gpt2_[1]_meco_window_cleaned.csv'
    word_data_filepath = f'data/processed/meco/words_en_df.csv'
    opt_dir = 'data/processed/meco/gpt2/optimization'
    compute_tensors = False
    pre_process = False
    norm_method = 'z-score'
    baselines='next_word,random'
    features = 'length,surprisal,frequency,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration'
    vectors_dir = 'data/processed/meco/gpt2'
    n_context_words = 7
    params_dataloader = {'batch_size': 32,
                         'shuffle': True,
                         'num_workers': 0}
    params_classifier = {'hidden_nodes': 128,
                         'output_nodes': n_context_words,
                         'learning_rate': 0.0001}
    epochs = 10
    seed = 42
    do_training = True
    do_feature_ablation = False

    # ------------------------------------------------------------

    eye_data = pd.read_csv(eye_data_filepath)
    word_data = pd.read_csv(word_data_filepath)

    if not os.path.isdir(opt_dir):
        os.mkdir(opt_dir)

    tensor_dir = opt_dir + f'/vectors_{features}'
    if not os.path.isdir(tensor_dir):
        os.mkdir(tensor_dir)

    # compute x and y tensors for each text if not computed and stored yet
    if compute_tensors:
        word_to_token_map = None
        if 'embedding' in features:
            with open(f'{vectors_dir}/word_to_token_map.pkl', 'rb') as f:
                word_to_token_map = pickle.load(f)
        convert_data_to_tensors(eye_data=eye_data, word_data=word_data, opt_dir=tensor_dir, level='word',
                                features=features, pre_process=pre_process, norm_method=norm_method,
                                data_filepath=eye_data_filepath, word_to_token_map=word_to_token_map,
                                vectors_dir=vectors_dir)

    # split at text level
    split_indices_test = split_data(eye_data['trialid'].unique(), split_type='train-test', test_size=.1, shuffle=True,
                                    random_state=seed,
                                    filepath=f'{opt_dir}/train_test_split.txt')
    train_eye_data = eye_data[eye_data['trialid'].isin(split_indices_test[0]['train_index'])].copy()
    split_indices = split_data(train_eye_data['trialid'].unique(), n_splits=5, shuffle=True, random_state=seed,
                               filepath=f'{opt_dir}/cross_val_splits.txt')

    # setting seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('Device: ', device)

    # start training
    print('Starting training...')

    if do_training:
        train_all(eye_data, split_indices, opt_dir, tensor_dir, params_dataloader, params_classifier, device, epochs, baselines)

    if do_feature_ablation:
        feature_ablation(eye_data, params_classifier, split_indices, device, opt_dir, tensor_dir, params_dataloader, epochs, baselines,
                         {'length': 0,
                                     'frequency': 2,
                                     'surprisal': 1,
                                     'has_been_fixated': 3,
                                     'previous_fix_duration': 1,
                                     'previous_sacc_distance': 0},
                         ['length,surprisal,frequency,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration',
                                         'frequency,surprisal,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration',
                                         'length,surprisal,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration',
                                         'length,frequency,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration',
                                         'length,frequency,surprisal,embedding,previous_sacc_distance,previous_fix_duration',
                                         'length,frequency,surprisal,has_been_fixated,previous_sacc_distance,previous_fix_duration',
                                         'length,frequency,surprisal,has_been_fixated,embedding,previous_fix_duration',
                                         'length,frequency,surprisal,has_been_fixated,embedding,previous_sacc_distance'],
                         ablation_type='mean')


if __name__ == '__main__':
    main()