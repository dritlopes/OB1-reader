import pandas as pd
import numpy as np
from typing import Literal
from copy import deepcopy
from fixations_to_words import fixation_to_word, WordLevelStatistics
from utils import from_csv_to_fixation_output
from model_components import FixationOutput
from utils import pre_process_string, extract_sentences
import re
import warnings

def to_dataframe(sim_word_stats: list[WordLevelStatistics]):

    """
    Convert a list of WordLevelStatistics to a pandas DataFrame.
    :param sim_word_stats: list of WordLevelStatistics from model simulation(s)
    :return: pandas DataFrame
    """
    return pd.DataFrame([obj.to_dict() for obj in sim_word_stats])

def match_eye_to_sim_columns(eye_data:pd.DataFrame,
                      corpus:Literal['provo', 'meco']):

    """
    Change the column names of the eye-tracking corpora to match the column names of the simulation data.
    :param eye_data: eye-tracking corpus dataframe.
    :param corpus: meco or provo.
    :return: eye-tracking corpus dataframe with changed column names.
    """
    if corpus == 'provo':
        mapping = {'IA_SKIP': 'skip',
                   'IA_LABEL': 'word',
                   'IA_ID': 'word_id',
                   'IA_REGRESSION_IN': 'incoming_regression',
                   'IA_REGRESSION_OUT': 'outgoing_regression',
                   'IA_FIRST_FIXATION_DURATION': 'first_fixation_duration',
                   'IA_FIRST_RUN_DWELL_TIME': 'gaze_duration',
                   'IA_DWELL_TIME': 'total_reading_time'}
        eye_data.rename(columns=mapping, inplace=True)

    elif corpus == 'meco':
        mapping = {"ia": "word",
                   "ianum": "word_id",
                   "reg.out": "outgoing_regression",
                   "reg.in": "incoming_regression",
                   "firsfix.dur": "first_fixation_duration",
                   "firstrun.dur": "gaze_duration",
                   "dur": "total_reading_time"}
        eye_data.rename(columns=mapping, inplace=True)

    else:
        raise NotImplementedError('Corpus not implemented. Choose between meco or provo.')

    return eye_data

def eval_stats(x,y):

    """
    Compute evaluation metrics (RMSE, corr, etc.) based on model simulated word measures and ground truth word measures from eye tracking data.
    :param x: simulated word measure
    :param y: true word measure
    :return: dict with metrics scores (RMSE, Pearson correlation, simulated mean, simulated standard deviation, true mean, true standard deviation)
    """

    rmse = np.sqrt(((x-y) ** 2).mean())
    corr = np.corrcoef(x, y)[0, 1]
    # print(f"\n RMSE={rmse}, mean_pred={np.mean(x)}, mean_true={np.mean(y)}, std_pred={np.std(x)}, std_true={np.std(y)}, corr={corr}\n")
    return {
        "RMSE": rmse,
        "corr": corr}

def clean_word(word: str):
    """
    Clean a word by removing non-alphanumeric characters and converting to lowercase.
    :param word: The word to clean.
    :return: The cleaned word.
    """
    return re.sub(r"[^a-zA-Z0-9]", "", word).lower()

def aggregate_and_align(eye_data:pd.DataFrame,
                        sim_data:pd.DataFrame,
                        corpus_name:str,
                        text_id:int,
                        stat_labels:list[str],
                        word_level_id_label:str,
                        word_content_label:str):

    """
    Aggregate eye-tracking data over participants for a specific text and align it with the simulated data from the model based on word IDs and word forms.
    :param eye_data: eye-tracking data from the corpus
    :param sim_data: simulated data by model on the same texts as in the eye-tracking data
    :param corpus_name: the name of the corpus (supported corpus: provo and meco).
    :param text_id: the ID of the text
    :param stat_labels: the labels for the statistics to aggregate on (e.g. skip, regression, etc)
    :param word_level_id_label: the column label for the word-level ID in the eye-tracking data
    :param word_content_label: the column label for the word form in the eye-tracking data
    :return:
    """

    # fix errors in the MECO data
    if corpus_name == 'meco' and text_id == 3:
        eye_data = eye_data[~((eye_data['ianum'] == 149) & (eye_data['uniform_id'].isin([f'en_{str(p)}' for p in
                                                                 [101, 102, 103, 3, 6, 72, 74, 76, 78, 79, 82, 83, 84,
                                                                  85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 97, 98,
                                                                  99]])))]
        eye_data['ianum'] = eye_data.apply(
            lambda x: x['ianum'] - 1 if (x['ianum'] >= 149) & (x['uniform_id'] in [f'en_{str(p)}' for p in
                                                                                   [101, 102, 103, 3, 6, 72, 74, 76, 78,
                                                                                    79, 82, 83, 84, 85, 86, 87, 88, 89,
                                                                                    90, 91, 93, 94, 95, 97, 98, 99]])
            else x['ianum'], axis=1)

        to_fix = (eye_data['uniform_id'].iloc[0] in [f'en_{str(p)}' for p in
                                               [101, 102, 103, 3, 6, 72, 74, 76, 78, 79, 82, 83, 84, 85, 86, 87, 88, 89,
                                                90, 91, 93, 94, 95, 97, 98, 99]])

        # align line number in the MECO simulation output
        if to_fix:
            sim_data = sim_data.drop(index=148, axis=0)

    # check consistency of eye-tracking word content for aggregation
    if not all(eye_data.groupby(word_level_id_label)[word_content_label].nunique() == 1):
        raise ValueError(
            f"Inconsistent values found in column '{word_content_label}' for the same '{word_level_id_label}'.")

    # aggregate eye-tracking data to word level
    agg_param = {k: "mean" for k in stat_labels}
    agg_param[word_content_label] = "first"
    eye_data = eye_data.groupby(word_level_id_label).agg(agg_param).reset_index()
    # print(eye_data.groupby("IA_ID")["IA_LABEL"].nunique().max()) # will print 1 if one word form per word id in the corpus, to check if keeping the first word form per word id is okay

    # check if words in eye-tracking data and simulated data are the same
    assert [clean_word(w) for w in eye_data[word_content_label]] == [clean_word(w) for w in sim_data[
        'word']], f"Words do not match: {eye_data[word_content_label]} vs {sim_data['word']}"

    sim_data['word_id'] = eye_data[word_level_id_label]

    return eye_data, sim_data

def compute_recognition_accuracy(simulation_output:pd.DataFrame):

    """
    Compute recognition accuracy based on the model's simulated output aggregated by word.
    :param simulation_output: Dataframe containing the simulated eye movements with a word column and a
    :return:
    """
    if "recognized_word_at_position" not in simulation_output.columns:
        warnings.warn("The column 'recognized_word_at_position' is missing from the simulation output. Not possible to compute recognition accuracy.")

    total_words = len(simulation_output)
    correct = [1 if word == recognized_word else 0 for word, recognized_word in zip(simulation_output["word"], simulation_output["recognized_word_at_position"])]
    accuracy = sum(correct) / total_words if total_words > 0 else 0

    return {"recognition_accuracy": accuracy}

def evaluate_reading (output:list[list[list[FixationOutput]]]|str,
                     corpus_name: Literal['provo', 'meco'] = 'provo',
                     data_filepath: str | None=None,
                     text_ids: list[int]=None,
                     eval_output_path = "../data/eval_output/eval_results.csv",
                     word_sim_output_path="../data/eval_output/agg_simulation.csv",
                     word_eye_output_path="../data/eval_output/agg_eye_tracking.csv",):

    """
    Evaluate reading performance based on eye-tracking data and simulated data.
    :param output: simulation output from model or path to saved simulation output.
    :param corpus_name: eye-tracking corpus name. Supported corpus: provo or meco.
    :param data_filepath: eye-tracking data file path. If None, default paths will be used for the specified corpus (meco or provo).
    :param text_ids: Specify which text ids from eye-tracking corpus were used for simulations if not all. If None, all texts will be used.
    :param eval_output_path: Path to save evaluation metrics results. Default is "../data/eval_output/eval_results.csv".
    :param word_sim_output_path: Path to save word-level model output. Default is "../data/eval_output/agg_simulation.csv".
    :param word_eye_output_path: Path to save word-level eye-tracking data averaged over participants. Default is "../data/eval_output/agg_eye_tracking.csv".
    :return:
    """

    if corpus_name == 'provo':
        if data_filepath is None:
            data_filepath = "../data/raw/Provo_Corpus-Eyetracking_Data.csv"
        text_level_id_label = "Text_ID"
        word_level_id_label = "IA_ID" # word level id
        word_content_label = "IA_LABEL"
        eye_mov_measures = ["IA_SKIP", "IA_REGRESSION_IN", "IA_REGRESSION_OUT", "IA_FIRST_FIXATION_DURATION", "IA_FIRST_RUN_DWELL_TIME", "IA_DWELL_TIME"]

    elif corpus_name == 'meco':
        if data_filepath is None:
            data_filepath = "../data/raw/joint_data_trimmed.csv"
        text_level_id_label = "itemid"
        word_level_id_label = "ianum"
        word_content_label = "ia"
        eye_mov_measures = ["skip", "reg.in", "reg.out", "firstfix.dur", "firstrun.dur", "dur"]

    else:
        raise NotImplementedError("Parameter `dataset` must be either `provo` or `meco`.")

    simulation_measures = ["skip", "outgoing_regression", "incoming_regression", "first_fixation_duration",
                           "gaze_duration", "total_reading_time"]

    if type(output) == str:
        output = from_csv_to_fixation_output(data_filepath=output)
    # output has first level number of simulations, then number of texts per simulation, then number of fixations per text.

    # get the texts from the eye mov data, based on the text_ids
    texts = extract_sentences(corpus_name, data_filepath, text_ids)

    # get list of tokens from each text from eye mov data
    list_of_tokens = []
    for text in texts:
        text_tokens = [pre_process_string(token) for token in text.split(' ')]
        list_of_tokens.append(text_tokens)

    # convert fixation level output from model to word level stats
    all_sim_word_stats = []
    for sim_n, list_of_fixations in enumerate(output):
        sim_word_stats = []
        for text_id, text_fixations in enumerate(list_of_fixations):
            text_tokens = list_of_tokens[text_id]
            text_word_stats = fixation_to_word(text_tokens, text_fixations)
            sim_word_stats.append(text_word_stats)
        all_sim_word_stats.append(sim_word_stats)

   # align simulated data to eye tracking data and compute evaluation metrics
    df = pd.read_csv(data_filepath)

    if corpus_name == 'meco':
        df = df[df['lang']=='en']

    all_sim_word_agg_dfs = [] # save data frames for each simulation to concatenate later
    for sim_id, simulation in enumerate(all_sim_word_stats):

        agg_sim, agg_eye_track = [], [] # save data frames for each text to concatenate later
        for (text_id, sim_data) in zip(text_ids, simulation):
            # for each word measure, aggregate eye-tracking data over participants (by mean)
            eye_data = deepcopy(df[df[text_level_id_label]==text_id])
            # normally WordLevelStatistics aggregates a word value across simulations, but here it just returns the same word value in a different format (no aggregation across simulations)
            # bcs we want to be able to evaluate each simulation against the eye-tracking data averaged over participants.
            sim_data_wls = [WordLevelStatistics([word]) for word in sim_data]
            for wls, word in zip(sim_data_wls, sim_data):
                wls.recognized_word_at_position = word.recognized_word_at_position
            sim_data = to_dataframe(sim_data_wls)
            eye_data, sim_data = aggregate_and_align(eye_data,
                                                     sim_data,
                                                     corpus_name,
                                                     text_id,
                                                     eye_mov_measures,
                                                     word_level_id_label,
                                                     word_content_label)
            sim_data["text_id"] = text_id
            sim_data["simulation_id"] = [sim_id for _ in range(len(sim_data))]
            eye_data = match_eye_to_sim_columns(eye_data, corpus_name)
            agg_sim.append(sim_data)
            eye_data["text_id"] = text_id
            agg_eye_track.append(eye_data)

        sim_df = pd.concat(agg_sim, axis=0, ignore_index=True)
        eye_df = pd.concat(agg_eye_track, axis=0, ignore_index=True)
        eye_df = eye_df[["text_id","word_id","word",*simulation_measures]]
        eye_df.to_csv(word_eye_output_path)
        # save simulation level aggregated data to list for later concatenation
        all_sim_word_agg_dfs.append(sim_df)

    # res_df = pd.DataFrame(eval_results)
    # res_df.to_csv(eval_output_path)
    sim_df = pd.concat(all_sim_word_agg_dfs, axis=0, ignore_index=True)
    sim_df = sim_df[["simulation_id","text_id","word_id","word",*simulation_measures, "recognized_word_at_position"]]
    sim_df.to_csv(word_sim_output_path)

    # compute eval metrics per simulation and per text
    eval_results_per_sim, eval_results_per_text, recog_acc_per_text = [], [], []

    for sim_id, sim_data in sim_df.groupby("simulation_id"):

        for label in simulation_measures:
            if label in eye_df.columns and label in sim_data.columns:
                result = eval_stats(sim_data[label].to_numpy(), eye_df[label].to_numpy())
                result["measure"] = label
                result["simulation_id"] = sim_id
                eval_results_per_sim.append(result)

        for text_id, text_data in sim_data.groupby("text_id"):
            recog_acc = compute_recognition_accuracy(text_data)
            recog_acc["text_id"] = text_id
            recog_acc["simulation_id"] = sim_id
            recog_acc_per_text.append(recog_acc)
            eye_text_data = eye_df[eye_df["text_id"]==text_id]

            for label in simulation_measures:
                if label in eye_text_data.columns and label in text_data.columns:
                    result = eval_stats(text_data[label].to_numpy(), eye_text_data[label].to_numpy())
                    result["measure"] = label
                    result["simulation_id"] = sim_id
                    result["text_id"] = text_id
                    eval_results_per_text.append(result)

    eval_results_per_sim = pd.DataFrame(eval_results_per_sim)
    eval_results_per_sim.to_csv(eval_output_path)
    eval_results_per_text = pd.DataFrame(eval_results_per_text)
    eval_results_per_text.to_csv(eval_output_path.replace(".csv", "_text_level.csv"))
    recog_acc_per_text = pd.DataFrame(recog_acc_per_text)
    recog_acc_per_text = recog_acc_per_text[['simulation_id','text_id','recognition_accuracy']]
    recog_acc_per_text.to_csv(eval_output_path.replace(".csv", "_recog_acc.csv"))

def evaluate_lex_decis(output, filename):
    #print(output.head)
    output['recog'] = output['recog RT'] > -1  # make column for recognition
    print(output.groupby('cond')['recog'].mean())
    outputRecogd = output.loc[output['recog RT'] > -1]      # select trials where there was recognition
    print(outputRecogd.groupby('cond')['recog RT'].mean())   # RTs for recog only when stim is recognized
    print(output.groupby('cond')['correct'].mean())
    print(output.groupby('cond')['LD RT'].mean())
    #print(output.groupby('cond')['av. max'].mean())
    print(output.groupby('cond')['av. tot'].mean())
    print(output.groupby('cond')['word_evidence'].mean())
    output.to_csv(filename)

# evaluate_reading(output="../data/model_output/_10_14_12-24-38/example_simulation.csv",
#                  corpus_name='provo',
#                  text_ids=[1])