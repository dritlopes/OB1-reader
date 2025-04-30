import pandas as pd
import numpy as np
from typing import Literal
from copy import deepcopy
import os
from fixations_to_words import fixation_to_word, WordLevelStatistics
from model_components import FixationOutput
from utils import pre_process_string

def to_meco_dict(wls: WordLevelStatistics):
    return {
        "ia": wls.word,
        "skip": wls.skip_rate,
        "reg.out": wls.outgoing_regression_rate,
        "reg.in": wls.incoming_regression_rate,
        "firstfix.dur": wls.mean_first_fixation_duration,
        "firstrun.dur": wls.mean_first_run_fixation_duration,
        "dur": wls.mean_total_reading_time,
    }

def to_provo_dict(wls: WordLevelStatistics):
    return {
        "Word": wls.word,
        "IA_SKIP": wls.skip_rate,
        "IA_REGRESSION_IN": wls.incoming_regression_rate,
        "IA_REGRESSION_OUT": wls.outgoing_regression_rate,
        "IA_FIRST_FIXATION_DURATION": wls.mean_first_fixation_duration,
        "IA_FIRST_RUN_DWELL_TIME": wls.mean_first_run_fixation_duration,
        "IA_DWELL_TIME": wls.mean_total_fixation_duration,
    }

def to_dataframe(wlss: list[WordLevelStatistics], format: Literal["provo", "meco"]):
    if format == "provo":
        return pd.DataFrame([to_provo_dict(obj) for obj in wlss])
    if format == "meco":
        return pd.DataFrame([to_meco_dict(obj) for obj in wlss])
    raise NotImplementedError()

def eval_stats(x,y,label):
    rmse = np.sqrt(((x-y) ** 2).mean())
    corr = np.corrcoef(x, y)[0, 1]
    # print(f"\n{label}: RMSE={rmse}, mean_pred={np.mean(x)}, mean_true={np.mean(y)}, std_pred={np.std(x)}, std_true={np.std(y)}, corr={corr}\n")
    return {
        "RMSE": rmse,
        "corr": corr,
        "mean_pred": np.mean(x),
        "std_pred": np.std(x),
        "mean_true": np.mean(y),
        "std_true": np.std(y)
    }

def extract_sentences(dataset:str, data_dir: str=None, text_ids: list[int]=None):
    if dataset == 'provo':
        if data_dir is None:
            data_dir = "../data/raw/Provo_Corpus-Eyetracking_Data.csv"
        text_level_id_label = "Text_ID"
        word_level_id_label = "IA_ID" # word level id
        word_content_label = "IA_LABEL"
        sep = ""
    elif dataset == 'meco':
        if data_dir is None:
            data_dir = "../data/raw/joint_data_trimmed.csv"
        text_level_id_label = "itemid"
        word_level_id_label = "ianum"
        word_content_label = "ia"
        sep = " "
    else:
        raise NotImplementedError("Parameter `dataset` must be either `provo` or `meco`.")
    texts = []
    df_original = pd.read_csv(data_dir)
    if dataset == 'meco':
        df_original = df_original[df_original['lang']=='en'] # TODO: support more languages
    if text_ids is None:
        if dataset == 'provo':
            text_ids = list(range(1,56)) # By default, there are sentences 1, 2, ..., 55 in Provo
        else:
            text_ids = list(range(1,13)) # By default, there are sentences 1, 2, ..., 12 in Meco
    for text_id in text_ids:
        df = deepcopy(df_original[df_original[text_level_id_label]==text_id])
        df = df.groupby(word_level_id_label).agg({
            word_content_label: "first"
        }).reset_index()
        texts.append(sep.join(list(df[word_content_label])).strip())

    return texts

def evaluate(output:list[list[list[FixationOutput]]], dataset: str = 'provo', data_dir: str|None=None, text_ids: list[int]=None, eval_output_path = "../data/eval_output/eval.csv", averaged_simulation_output_path="../data/eval_output/simulation.csv"):
    if dataset == 'provo':
        if data_dir is None:
            data_dir = "../data/raw/Provo_Corpus-Eyetracking_Data.csv"
        text_level_id_label = "Text_ID"
        word_level_id_label = "IA_ID" # word level id
        word_content_label = "IA_LABEL"
        stat_labels = ["IA_SKIP", "IA_REGRESSION_IN", "IA_REGRESSION_OUT", "IA_FIRST_FIXATION_DURATION", "IA_FIRST_RUN_DWELL_TIME", "IA_DWELL_TIME"]
    elif dataset == 'meco':
        if data_dir is None:
            data_dir = "../data/raw/joint_data_trimmed.csv"
        text_level_id_label = "itemid"
        word_level_id_label = "ianum"
        word_content_label = "ia"
        stat_labels = ["skip", "reg.in", "reg.out", "firstfix.dur", "firstrun.dur", "dur"]
    else:
        raise NotImplementedError("Parameter `dataset` must be either `provo` or `meco`.")

    texts = extract_sentences(dataset, data_dir, text_ids)
    
    list_of_tokens = []
    for text in texts:
        text_tokens = [pre_process_string(token) for token in text.split(' ')]
        list_of_tokens.append(text_tokens)
    word_level_stats = [[[] for token in tokens] for tokens in list_of_tokens]
    for list_of_fixations in output:
        for i,(tokens, fixations) in enumerate(zip(list_of_tokens, list_of_fixations)):
            wlos = fixation_to_word(tokens, fixations)
            for j,wlo in enumerate(wlos):
                word_level_stats[i][j].append(wlo)
    wlss = [[WordLevelStatistics(w) for w in ws] for ws in word_level_stats]

    df = pd.read_csv(data_dir)
    if dataset == 'meco':
        df = df[df['lang']=='en']
    sim_results = []
    results = []
    for (text_id, wls) in zip(text_ids, wlss):
        gt = deepcopy(df[df[text_level_id_label]==text_id])
        agg_param = {k:"mean" for k in stat_labels}
        agg_param[word_content_label] = "first"
        gt = gt.groupby(word_level_id_label).agg(agg_param).reset_index()
        sim = to_dataframe(wls, dataset)
        sim[text_level_id_label] = text_id
        sim_results.append(sim)
    
        for label in stat_labels:
            result = eval_stats(sim[label], gt[label], label)
            result[text_level_id_label] = text_id
            result["Label"] = label
            results.append(result)
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(averaged_simulation_output_path), exist_ok=True)
    res_df = pd.DataFrame(results) #returns evaluation statistic results
    res_df.to_csv(eval_output_path)
    sim_df = pd.concat(sim_results, axis=0, ignore_index=True) #returns simulation results
    sim_df.to_csv(averaged_simulation_output_path)