import pandas as pd
import os
import numpy as np
import chardet
import json
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from model_components import semantic_processing
from collections import defaultdict
from typing import Literal
import codecs

def pre_process_string(string, remove_punctuation=True, all_lowercase=True, strip_spaces=True):

    """
    Format tokens from input text to be the same as in the model's lexicon.

    :param string: token from tokenized text input.
    :param remove_punctuation: remove punctuation from token.
    :param all_lowercase: lowercase token.
    :param strip_spaces: remove trailing spaces from token.
    :return: formatted token.
    """

    if remove_punctuation:
        string = re.sub(r'[^\w\s]', '', string)
    if all_lowercase:
        string = string.lower()
    if strip_spaces:
        string = string.strip()

    return string

def create_freq_dict(language:str, task_words:list[str]|set[str], freq_threshold = 1, n_high_freq_words = 500, verbose = False):

    """
    Create a dictionary that maps each word to its frequency.

    :param language: language input text(s) are in.
    :param task_words: words from input text(s).
    :param freq_threshold: above what frequency value a word needs to be to get included.
    :param n_high_freq_words: number of top frequent words to include from frequency resource.
    :param verbose: whether to print out progress messages.
    :return: word frequency dictionary.
    """

    if language == 'english':
        filepath = '../data/raw/SUBTLEX_UK.txt'
        columns_to_use = [0, 1, 5]
        freq_type = 'LogFreq(Zipf)'
        word_col = 'Spelling'

    elif language == 'french':
        filepath = '../data/raw/French_Lexicon_Project.txt'
        columns_to_use = [0, 7, 8, 9, 10]
        freq_type = 'cfreqmovies'
        word_col = 'Word'

    elif language == 'german':
        filepath = '../data/raw/SUBTLEX_DE.txt'
        columns_to_use = [0, 1, 3, 4, 5, 9]
        freq_type = 'lgSUBTLEX'
        word_col = 'Word'

    elif language == 'dutch':
        filepath = '../data/raw/SUBTLEX-NL.txt' # first delete rows with noisy encoding! otherwise encoding error
        columns_to_use = [0, 7]
        freq_type = 'Zipf'
        word_col = 'Word'

    else:
        raise NotImplementedError(language + " is not implemented yet. Please choose between English, French, German or Dutch.")

    # create dict of word frequencies from resource/corpus file
    freq_df = pd.read_csv(filepath, usecols=columns_to_use, dtype={word_col: np.dtype(str)},
                          encoding=chardet.detect(open(filepath, "rb").read())['encoding'], delimiter="\t")
    freq_df.sort_values(by=[freq_type], ascending=False, inplace=True, ignore_index=True)
    freq_df[word_col] = freq_df[word_col].astype('unicode')
    freq_words = freq_df[[word_col, freq_type]].copy()
    # NV: convert to Zipf scale. # from frequency per million to zipf. Also, replace -inf with 1
    if freq_type == 'cfreqmovies':
        freq_words['cfreqmovies'] = freq_words['cfreqmovies'].apply(lambda x: np.log10(x * 1000) if x > 0 else 0)
    # convert strings to floats
    freq_words[freq_type] = freq_words[freq_type].replace(',', '.', regex=True).astype(float)
    # preprocess words for correct matching with tokens in stimulus
    freq_words[word_col] = freq_words[word_col].apply(lambda x : pre_process_string(x))
    # only keep words whose frequencies are higher than threshold
    # frequencies = freq_words[freq_type].tolist()
    # print(f'Max frequency value: {max(frequencies)}, word "{freq_words[word_col].tolist()[frequencies.index(max(frequencies))]}"')
    # print(f'Min frequency value: {min(frequencies)}, word "{freq_words[word_col].tolist()[frequencies.index(min(frequencies))]}"')
    freq_words = freq_words[freq_words[freq_type] > freq_threshold]
    frequency_words_dict = dict(zip(freq_words[freq_words.columns[0]], freq_words[freq_words.columns[1]]))

    # create dict with frequencies from words in task stimuli which are also in the resource/corpus file
    file_freq_dict = {}
    overlapping_words = []

    if task_words:
        overlapping_words = list(set(task_words) & set(freq_words[word_col].tolist()))
        for word in overlapping_words:
            file_freq_dict[word] = frequency_words_dict[word]
    # add top n words from frequency resource
    if n_high_freq_words:
        for word, freq in zip(freq_words[word_col].tolist()[:n_high_freq_words], freq_words[freq_type].tolist()[:n_high_freq_words]):
            # file_freq_dict[(freq_words.iloc[line_number][0])] = freq_words.iloc[line_number][1]
            file_freq_dict[word] = freq

    elif not task_words and not n_high_freq_words:
        raise AttributeError('Provide either n_high_freq_words or task_words to get_word_freq function.')

    if verbose:
        if task_words:
            print(f"amount of unique words in input: {len(set(task_words))}")
            print(f'words not in frequency resource: {set(task_words).difference(set(freq_words[word_col].tolist()))}')
            if overlapping_words:
                print(f"amount of unique input words in frequency resource: {len(overlapping_words)}")
        if n_high_freq_words:
            print(f"Including top {n_high_freq_words} most frequent words: {len(file_freq_dict.keys())}")

    return file_freq_dict

def get_word_freq(words:list[str]|set[str]=None, language:str='english', output_word_frequency_map='', n_high_freq_words = 500, freq_threshold = 1, word_predictability=None, verbose=True)->dict:

    """
    Provide dictionary with words as keys and frequencies as values.

    :param words: words from input texts which need frequency values.
    :param language: the language the input text is in.
    :param output_word_frequency_map: filepath to load existent word frequency map, or to save out frequency map if it does not exist yet.
    :param n_high_freq_words: number of top frequent words to include from frequency resource.
    :param freq_threshold: above what frequency value a word needs to be to get included.
    :param word_predictability: give dict with word predictability to arguments if predicted words should also have frequency values to be used in the activation computation by the model
    :param verbose: whether to print out progress messages.
    :return: dict with words as keys and frequencies as values.
    """

    # AL: if filepath is given and if frequency map exists there, just read it in
    if output_word_frequency_map and os.path.exists(output_word_frequency_map):
        with open(output_word_frequency_map, "r") as f:
            word_freq_dict = json.load(f)
    # AL: if filepath is given, but frequency map does not exist, OR no filepath is given
    else:
        word_freq_dict = create_freq_dict(language, words, freq_threshold,
                                         n_high_freq_words, verbose)
        # AL: predicted tokens need frequency values to be added to lexicon
        if word_predictability:
            predicted_tokens = return_predicted_tokens(word_predictability)
            if verbose:
                print(f"Adding predicted words to frequency dictionary...")
            frequency_predicted_tokens = create_freq_dict(language, predicted_tokens, n_high_freq_words=0, verbose=True)
            word_freq_dict.update(frequency_predicted_tokens)
            if verbose:
                print(f"{len(set(frequency_predicted_tokens.keys()).difference(set(word_freq_dict.keys())))} words were added to frequency dictionary.")
        else:
            if verbose:
                print("No predicted words were found to add to frequency dictionary. Using only words from input text(s) and/or top most frequent words from frequency resource.")
        # AL: save out frequency dict
        if not output_word_frequency_map:
            output_word_frequency_map = f"../data/processed/frequency_map_{language}.json"
        with open(output_word_frequency_map, "w") as f:
            json.dump(word_freq_dict, f, ensure_ascii=False)
        print(f'frequency file stored in {output_word_frequency_map}')

    return word_freq_dict


def create_pred_file(texts:list[str], stimulus_name:str, language:str, pred_source:str, topk:int, pred_threshold:float, model_token:str)->dict:

    """
    Create dictionary with predictability values.

    :param texts: list of texts to be processed by model.
    :param stimulus_name: name of set of input texts. E.g. name of the corpus.
    :param language: language of input texts.
    :param pred_source: name of resource used to get predictability.
    :param topk: number of top predictions to use. If 'all', all predictions (above threshold) are returned.
    :param pred_threshold: the minimum prediction score for word to be included.
    :param model_token: token to allow access to llama model. Specific for when generating predictions with llama.
    :return: dictionary with predictability values.
    """

    word_pred_values_dict = dict()

    if pred_source.lower() in ['gpt2', 'llama']:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device ', str(device))

        if language == 'english':

            # load language model and its tokenizer
            if 'gpt2' in pred_source.lower():
                language_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
                lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            elif 'llama' in pred_source.lower():
                lm_tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=model_token)
                language_model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token=model_token, torch_dtype=torch.float16).to(
                    device)

            # Additional info when using cuda
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

            # list of words, set of words, sentences or passages. Each one is equivalent to one trial in an experiment
            for i, sequence in enumerate(texts):
                sequence = [token for token in sequence.split(' ') if token != '']
                pred_dict = dict()
                pred_info = semantic_processing(sequence, lm_tokenizer, language_model, pred_source, topk, pred_threshold, device)
                for pos in range(1, len(sequence)):
                    target = pre_process_string(sequence[pos])
                    pred_dict[str(pos)] = {'target': target,
                                            'predictions': dict()}
                    for token, pred in zip(pred_info[pos][0], pred_info[pos][1]):
                        token_processed = pre_process_string(token)
                        # language models may use uppercase, while our lexicon only has lowercase, so take the higher pred
                        if token_processed not in pred_dict[str(pos)]['predictions'].keys():
                            pred_dict[str(pos)]['predictions'][token_processed] = pred

                word_pred_values_dict[str(i)] = pred_dict

        else:
            raise NotImplementedError('Language not implemented yet.')

    elif pred_source == 'cloze':

        if stimulus_name:

            if 'psc' in stimulus_name.lower():
                filepath = "../data/raw/PSCall_freq_pred.txt"
                my_data = pd.read_csv(filepath, delimiter="\t",
                                      encoding=chardet.detect(open(filepath, "rb").read())['encoding'])
                word_pred_values_dict = np.array(my_data['pred'].tolist())

            elif 'provo' in stimulus_name.lower():
                # encoding = chardet.detect(open(filepath, "rb").read())['encoding']
                filepath = "../data/raw/Provo_Corpus-Predictability_Norms.csv"
                my_data = pd.read_csv(filepath, encoding="ISO-8859-1")
                # align indexing with ob1 stimuli (which starts at 0, not at 1)
                my_data['Text_ID'] = my_data['Text_ID'].apply(lambda x : str(int(x)-1))
                my_data['Word_Number'] = my_data['Word_Number'].apply(lambda x : str(int(x)-1))
                # fix misplaced row in raw data
                my_data.loc[(my_data['Word_Number'] == '2') & (my_data['Word'] == 'probably') & (my_data['Text_ID'] == '17'), 'Text_ID'] = '54'

                for text_id, info in my_data.groupby('Text_ID'):

                    word_pred_values_dict[text_id] = dict()

                    for text_position, responses in info.groupby('Word_Number'):

                        # fix error in provo cloze data indexing
                        for row in [(2,44),(12,18)]:
                            if int(text_id) == row[0] \
                                    and int(text_position) in range(row[1]+1, len(info['Word_Number'].unique())+2):
                                text_position = str(int(text_position) - 1)

                        target = pre_process_string(responses['Word'].tolist()[0])
                        word_pred_values_dict[text_id][text_position] = {'target': target,
                                                                         'predictions': dict()}
                        responses = responses.to_dict('records')
                        for response in responses:
                            if response['Response'] and type(response['Response']) == str:
                                word = pre_process_string(response['Response'])
                                word_pred_values_dict[text_id][text_position]['predictions'][word] = float(response['Response_Proportion'])
            else:
                raise NotImplementedError('Resource given for cloze predictability is not implemented yet. '
                                          'Choose between "provo" and "psc", and make sure the resource is respectively located in "../data/raw/Provo_Corpus-Predictability_Norms.csv"'
                                          'and "../data/raw/PSCall_freq_pred.txt"')
        else:
            raise NotImplementedError('Provide the name of the resource where cloze predictability is located. E.g. provo')
    else:
        raise NotImplementedError('Prediction source is not implemented yet. Please choose between gpt2, llama2, or cloze (from Provo corpus or PSC corpus).')

    return word_pred_values_dict

def get_word_pred(texts:[list[str]]=None, language:str='english', output_word_pred_map='', pred_source='gpt2', stimulus_name:str='', topk='all', pred_threshold=0.01, model_token='')->dict:

    """
    Create dictionary with predictability values for predicted words, given input texts.

    :param output_word_pred_map: filepath to save predictability values.
    :param texts: list of text words to be processed by model.
    :param language: language of input texts.
    :param stimulus_name: name of set of input texts. E.g. name of the corpus.
    :param pred_source: name of resource used to get predictability.
    :param topk: number of top predictions to use.
    :param pred_threshold: the minimum predictability value for word to be included.
    :param model_token: token to allow access to llama model. Specific for when generating predictions with llama.
    :return: dictionary with predictability values.
    """

    # AL: if filepath is given and if predictability map exists there, just read it in
    if output_word_pred_map and os.path.exists(output_word_pred_map):
        with open(output_word_pred_map, "r") as f:
            word_pred_dict = json.load(f)
    # AL: if filepath is given, but predictability map does not exist, OR no filepath is given
    else:
        word_pred_dict = create_pred_file(texts, stimulus_name, language, pred_source, topk, pred_threshold, model_token)
        # AL: save out predictability map
        if not output_word_pred_map:
            output_word_pred_map = f"../data/processed/prediction_map_{pred_source}_{language}.json"
            if pred_source in ['gpt2', 'llama']:
                output_word_pred_map = output_word_pred_map.replace('.json', f'_topk{topk}_{pred_threshold}.json')
        # AL: save out pred map
        with open(output_word_pred_map, "w") as f:
            json.dump(word_pred_dict, f, ensure_ascii=False)

    return word_pred_dict

def return_predicted_tokens(word_pred_dict:dict)->set:

    """
    Return predicted types (unique tokens) from given predictability dictionary.

    :param word_pred_dict: the dictionary with predicted words as keys and prediction scores as values.
    :return: predicted types.
    """

    types = set()

    if word_pred_dict:

        for text_id, text_pred in word_pred_dict.items():
            for pos, pos_info in text_pred.items():
                for token, pred_value in pos_info['predictions'].items():
                    if token:
                        types.add(token)

    return types

def write_out_simulation_data(simulation_data:list[list]|list, outfile_sim_data:str, simulation_id=None, text_id=None):

    """
    Write out simulation data to file.

    :param simulation_data: list of FixationOutput objects. Each object contains the data of one simulated fixation.
    :param outfile_sim_data: filepath to save simulation data.
    :param simulation_id: index of simulation which output is being saved.
    :param text_id: index of text which output is being saved.
    """

    # if saving output at text level (with every text processed, save out output)
    if simulation_id is not None and text_id is not None:

        simulation_results = defaultdict(list)

        for fix_counter, fix_info in enumerate(simulation_data):
            simulation_results['fixation_counter'].append(fix_counter)
            simulation_results['text_id'].append(text_id)
            simulation_results['simulation_id'].append(simulation_id)
            if type(fix_info) != dict():
                # convert object fields to dict keys and values
                fix_info = fix_info.to_dict()
            for info_name, info_value in fix_info.items():
                simulation_results[info_name].append(info_value)
        simulation_results_df = pd.DataFrame.from_dict(simulation_results)

        # if this is not the first text of first simulation, read in output of previous texts/simulations and update it with new data
        if os.path.exists(outfile_sim_data):
            simulation_df = pd.read_csv(outfile_sim_data)
            pd.concat([simulation_df, simulation_results_df])

        # save out output
        dir_path = os.path.dirname(outfile_sim_data)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        simulation_results_df.to_csv(outfile_sim_data, sep='\t', index=False)

    # if simulation id not give, save output above simulation level (save all outputs at once, once all simulations in all input texts are done)
    else:

        simulation_results = defaultdict(list)
        for sim_index, texts_simulations in enumerate(simulation_data):
            for text_index, text_fixations in enumerate(texts_simulations):
                for fix_counter, fix_info in enumerate(text_fixations):
                    simulation_results['fixation_counter'].append(fix_counter)
                    simulation_results['text_id'].append(text_index)
                    simulation_results['simulation_id'].append(sim_index)
                    if type(fix_info) != dict():
                        # convert object fields to dict keys and values
                        fix_info = fix_info.to_dict()
                    for info_name, info_value in fix_info.items():
                        simulation_results[info_name].append(info_value)
        simulation_results_df = pd.DataFrame.from_dict(simulation_results)
        dir_path = os.path.dirname(outfile_sim_data)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        simulation_results_df.to_csv(outfile_sim_data, sep='\t', index=False)

def get_ngram_frequency_from_file(filepath, sep='\t'):

    # check if filepath exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist trying to get ngram frequencies from file.")

    else:
        data = pd.read_csv(filepath, sep=sep)

    # assert that column 'bigram' and column 'freq' exist
    assert 'bigram' in data.columns and 'freq' in data.columns, f"Columns 'bigram' and 'freq' must exist in the file {filepath}."

    return data