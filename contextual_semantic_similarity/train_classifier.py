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

# nn class definition
class Classifier(nn.Module):

    # initialise the neural network
    def __init__(self,
                 input_nodes,
                 hidden_nodes,
                 output_nodes,
                 learning_rate=0.001):
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

        # set the number of nodes in the input, hidden and output layers and the learning rate.
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # define layers
        self.layer_stack = nn.Sequential(nn.Linear(input_nodes, hidden_nodes),
                                         nn.ReLU(),
                                         nn.Linear(hidden_nodes, hidden_nodes),
                                         nn.ReLU(),
                                         nn.Linear(hidden_nodes, output_nodes))

    def forward(self, inputs):

        logits = self.layer_stack(inputs)

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

        for batch_x, batch_y in training_dataset:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass (model outputs raw logits)
            y_logits = model(batch_x)

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

        for val_batch_x, val_batch_y in val_dataset:

            with torch.inference_mode():

                val_batch_x, val_batch_y = val_batch_x.to(device), val_batch_y.to(device)

                val_logits = model(val_batch_x)
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

        for test_x, test_y in test_dataset:

            test_x, test_y = test_x.to(device), test_y.to(device)
            test_logits = model(test_x)
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
                    value = df[df['Unnamed: 0'] == measure]['macro avg'].tolist()[0]
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
              features:str,
              params_dataloader, params_classifier, device, epochs, baselines):

    all_targets, all_predictions, all_models = [], [], []

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
        params_classifier['input_nodes'] = training_set.x_tensor.shape[1]
        params_classifier['hidden_nodes'] = training_set.x_tensor.shape[1]
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
            report.to_csv(f'{opt_dir}/report_split{i}_baseline_{baseline}.csv')
            cf_matrix.to_csv(f'{opt_dir}/confusion_matrix_split{i}_baseline_{baseline}.csv')
            all_targets.append(y_true)
            all_predictions.append(y_pred)
            all_models.extend([baseline for i in range(y_true.shape[0])])

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    print('Creating graphs of validation results...')
    display_prediction_distribution(all_targets, all_predictions,
                                    filepath=f'{opt_dir}/distribution_nn_all_val_splits.tiff', col=all_models)
    all_values, all_models, all_measures, _ = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir=opt_dir,
                                                             measures='f1-score,accuracy', models='model,' + baselines)
    display_eval(all_values, all_models, all_measures, filepath=f'{opt_dir}/eval.tiff')
    test_sig_diff(all_values, all_models, all_measures, opt_dir)

def feature_ablation(eye_data, params_classifier, split_indices, device, opt_dir, tensor_dir, params_dataloader, epochs,
                     baselines, all_features, features_to_select):

    print('Feature ablation...')
    all_targets, all_predictions, all_models = [], [], []

    for feature_combi in features_to_select:

        print(f'features: {feature_combi}')

        for i, split in enumerate(split_indices):
            train_split_ids = [f'{participant_id},{text_id}' for text_id in split['train_index']
                               for participant_id in
                               eye_data[eye_data['trialid'] == text_id]['participant_id'].unique()]
            training_set = FixationDataset(train_split_ids, tensor_dir, features_to_select=feature_combi, all_features=all_features)
            training_generator = DataLoader(training_set, **params_dataloader)
            val_split_ids = [f'{participant_id},{text_id}' for text_id in split['test_index']
                             for participant_id in eye_data[eye_data['trialid'] == text_id]['participant_id'].unique()]
            val_set = FixationDataset(val_split_ids, tensor_dir, features_to_select=feature_combi, all_features=all_features)
            val_generator = DataLoader(val_set, **params_dataloader)

            # loss_weights = compute_class_weight("balanced",
            #                                     classes=np.unique(training_set.y_tensor),
            #                                     y=np.array(training_set.y_tensor))
            # loss_weights = torch.FloatTensor(loss_weights).to(device)

            params_classifier['input_nodes'] = training_set.x_tensor.shape[1]
            params_classifier['hidden_nodes'] = training_set.x_tensor.shape[1]
            model = Classifier(**params_classifier).to(device)
            train_model(model=model,
                        training_dataset=training_generator,
                        val_dataset=val_generator,
                        epochs=epochs,
                        device=device,
                        n_split=i,
                        loss_weights=None)

            torch.save(model, f'{opt_dir}/classifier_split{i}_{feature_combi}.pth')
            # model = torch.load(f'{opt_dir}/classifier_split{i}_{feature_combi}.pth', map_location=device)

            y_pred, y_true = test_model(model=model, test_dataset=val_generator, device=device)
            y_true = y_true - 3
            y_pred = y_pred - 3
            report, cf_matrix = evaluate_model(y_true, y_pred)
            report.to_csv(f'{opt_dir}/report_split{i}_{feature_combi}.csv')
            cf_matrix.to_csv(f'{opt_dir}/confusion_matrix_split{i}_{feature_combi}.csv')
            all_targets.append(y_true)
            all_predictions.append(y_pred)
            all_models.extend([feature_combi for i in range(y_true.shape[0])])

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    display_prediction_distribution(all_targets, all_predictions,
                                    filepath=f'{opt_dir}/distribution_nn_feature_ablation_all_val_splits_1.tiff',
                                    col=all_models)
    features_to_select.extend(baselines.split(','))
    all_values, all_err_models, all_measures, _ = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir=opt_dir,
                                                                 measures='f1-score,accuracy',
                                                                 models= features_to_select)
    display_eval(all_values, all_err_models, all_measures, filepath=f'{opt_dir}/eval_feature_ablation_1.tiff')

def train_per_participant(eye_data, split_indices, tensor_dir, features, params_dataloader, device, epochs,
                          params_classifier, opt_dir, baselines):

    # # sample participants (and make sure all participants have read all texts in the train and val splits)
    # p = eye_data['participant_id'].unique().tolist()
    # participant_set = random.sample(p, 5)
    # for p in participant_set:
    #     text_ids = eye_data[eye_data['participant_id'] == p]['trialid'].unique()
    #     for text_id in split_indices_test[0]['train_index']:
    #         if text_id not in text_ids:
    #             raise ValueError(f'Participant {p} has not read text {text_id}.')
    # print(participant_set)
    # for p, data in eye_data.groupby('participant_id'):
    #     print(p)
    #     print(data['trialid'].unique())

    participant_set = 'en_3,en_6,en_72,en_74,en_93'  # the first to have read all texts

    for participant_id in participant_set.split(','):

        all_targets, all_predictions, all_models = [], [], []
        print(f'Participant: {participant_id}')
        participant_data = eye_data[eye_data['participant_id'] == participant_id].copy()

        for i, split in enumerate(split_indices):

            print(f'Split: {i}')

            # Prepare data
            train_split_ids = [f'{participant_id},{text_id}' for text_id in split['train_index'] if
                               text_id in participant_data['trialid'].unique()]
            training_set = FixationDataset(train_split_ids, tensor_dir, features=features)
            training_generator = DataLoader(training_set, **params_dataloader)

            test_split_ids = [f'{participant_id},{text_id}' for text_id in split['test_index'] if
                              text_id in participant_data['trialid'].unique()]
            val_set = FixationDataset(test_split_ids, tensor_dir, features=features)
            val_generator = DataLoader(val_set, **params_dataloader)

            # Train model
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
            torch.save(model, f'{opt_dir}/classifier_split{i}_participant{participant_id}.pth')
            # model = torch.load(f'{opt_dir}/classifier_split{i}_participant{participant_id}.pth', map_location=device)

            # Evaluate model with validation set
            print('Testing trained model...')
            y_pred, y_true = test_model(model=model, test_dataset=val_generator, device=device)
            # convert labels back to -3 to +3
            y_true = y_true - 3
            y_pred = y_pred - 3
            report, cf_matrix = evaluate_model(y_true, y_pred)
            report.to_csv(f'{opt_dir}/report_split{i}_participant{participant_id}.csv')
            cf_matrix.to_csv(f'{opt_dir}/confusion_matrix_split{i}_participant{participant_id}.csv')
            all_targets.append(y_true)
            all_predictions.append(y_pred)
            all_models.extend(['model' for i in range(y_true.shape[0])])

            # Evaluate baselines
            print('Computing and evaluating baselines...')
            for baseline in baselines.split(','):
                if baseline in ['next_word', '7letter_2right']:
                    y_pred = load_baseline_tensors(test_split_ids, baseline, tensor_dir)
                    y_true = val_set.y_tensor
                    y_true, y_pred = clean_tensors(y_true, y_pred)
                    y_true = y_true - torch.tensor(3)
                    y_pred = y_pred.detach().cpu().numpy()
                    y_true = y_true.detach().cpu().numpy()
                else:
                    print('Training random model...')
                    model = Classifier(**params_classifier).to(device)
                    training_set = FixationDataset(train_split_ids, tensor_dir, features=features, random=True)
                    training_generator = DataLoader(training_set, **params_dataloader)
                    val_set = FixationDataset(test_split_ids, tensor_dir, features=features, random=True)
                    val_generator = DataLoader(val_set, **params_dataloader)
                    train_model(model=model,
                                training_dataset=training_generator,
                                val_dataset=val_generator,
                                epochs=epochs,
                                device=device,
                                n_split=i,
                                loss_weights=None)
                    torch.save(model, f'{opt_dir}/classifier_split{i}_participant{participant_id}_random.pth')
                    y_pred, y_true = test_model(model=model, test_dataset=val_generator, device=device)
                    y_true = y_true - 3
                    y_pred = y_pred - 3
                report, cf_matrix = evaluate_model(y_true, y_pred)
                report.to_csv(f'{opt_dir}/report_split{i}_baseline_{baseline}_participant{participant_id}.csv')
                cf_matrix.to_csv(
                    f'{opt_dir}/confusion_matrix_split{i}_baseline_{baseline}_participant{participant_id}.csv')
                all_targets.append(y_true)
                all_predictions.append(y_pred)
                all_models.extend([baseline for i in range(y_true.shape[0])])

        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        display_prediction_distribution(all_targets, all_predictions,
                                        filepath=f'{opt_dir}/distribution_nn_all_val_splits_participant{participant_id}.tiff',
                                        col=all_models,
                                        title=participant_id)
        all_values, all_models, all_measures, _ = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir=opt_dir,
                                                                 measures='f1-score,accuracy',
                                                                 models='model,' + baselines,
                                                                 participant_ids=participant_id)
        display_eval(all_values, all_models, all_measures, filepath=f'{opt_dir}/eval_{participant_id}.tiff')

    all_values, all_models, all_measures, all_participants = read_in_scores(splits=[0, 1, 2, 3, 4], opt_dir=opt_dir,
                                                                            measures='f1-score,accuracy',
                                                                            models='model,' + baselines,
                                                                            participant_ids=participant_set)
    display_eval(all_values, all_models, all_measures, filepath=f'{opt_dir}/eval_per_participant.tiff',
                 col=all_participants, col_name='participant')

def main():

    eye_data_filepath = f'data/processed/meco/gpt2/full_gpt2_[1]_meco_window_cleaned.csv'
    word_data_filepath = f'data/processed/meco/words_en_df.csv'
    opt_dir = 'data/processed/meco/gpt2/optimization'
    compute_tensors = False
    pre_process = False
    norm_method = 'z-score'
    baselines='next_word,random'
    features = 'similarity,length,entropy,surprisal,frequency,previous_sacc_distance,previous_fix_duration'
    n_context_words = 7
    params_dataloader = {'batch_size': 32,
                         'shuffle': True,
                         'num_workers': 0}
    params_classifier = {'output_nodes': n_context_words,
                         'learning_rate': 0.0001}
    epochs = 10
    seed = 42
    do_training = False
    do_feature_ablation = True
    do_training_per_participant = False

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
        convert_data_to_tensors(eye_data, word_data, tensor_dir,
                                'word', features, pre_process,
                                norm_method, eye_data_filepath)

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
        train_all(eye_data, split_indices, opt_dir, tensor_dir, features, params_dataloader, params_classifier, device, epochs, baselines)

    if do_feature_ablation:
        feature_ablation(eye_data, params_classifier, split_indices, device, opt_dir, tensor_dir, params_dataloader, epochs,
                         baselines, features,
                         ['length,previous_sacc_distance,previous_fix_duration', 'length,frequency,previous_sacc_distance,previous_fix_duration',
                          'length,surprisal,frequency,previous_sacc_distance,previous_fix_duration', 'length,surprisal,frequency',
                          'length,surprisal,frequency,previous_sacc_distance', 'length,surprisal,frequency,previous_sacc_distance,previous_fix_duration'])

    if do_training_per_participant:
        train_per_participant(eye_data, split_indices, tensor_dir, features, params_dataloader, device, epochs,
                              params_classifier, opt_dir, baselines)

if __name__ == '__main__':
    main()