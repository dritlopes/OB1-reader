import logging
import numpy as np
import torch
from torch import nn
import math
import warnings
from reading_helper_functions import string_to_open_ngrams, cal_ngram_exc_input, is_similar_word_length, \
    get_midword_position_for_surrounding_word, calc_word_attention_right, calc_saccade_error, define_slot_matching_order, \
    find_word_edges, sample_from_norm_distribution

logger = logging.getLogger(__name__)

def compute_stimulus(fixation:int, tokens:list[str], stimulus_window:str)->(str,list,int):

    """
    Given fixation position in text and the text tokens, find the stimulus for a given fixation.

    :param fixation: which token from input text the fixation is located at.
    :param tokens: the text tokens.
    :param stimulus_window: which positions around the fixated word the model should process in parallel.

    :return: the stimulus, the position of each word in the stimulus in relation to the text,
    and the position of the fixated word in relation to the stimulus.
    """

    stimulus_window = stimulus_window.split(',')
    start_window = fixation + int(stimulus_window[0])
    end_window = fixation + int(stimulus_window[1])
    # only add position if after text begin and below text length
    stimulus_position = [i for i in range(start_window, end_window+1) if i >= 0 and i < len(tokens)]
    stimulus = ' '.join([tokens[i] for i in stimulus_position])
    fixated_position_stimulus = stimulus_position.index(fixation)

    return stimulus, stimulus_position, fixated_position_stimulus

def compute_eye_position(stimulus:str, fixated_position_stimulus:int, eye_position:int=None)->int:

    """
    Based on the stimulus, the position of the fixated word in the stimulus, and the position of the fixation center within the fixated word,
    find the position of the center of fixation in relation to the stimulus.

    :param stimulus: the tokens the model is processing in parallel (during one fixation).
    :param fixated_position_stimulus: the position of the fixed word in stimulus.
    :param eye_position: which letter is the center of fixation in the fixated word.
    Given the stimulus during a fixation, find where the eye is positioned in relation to the stimulus.

    :return: the index of the character the eyes are fixating at in the stimulus (in number of characters).
    """

    if eye_position is None:
        stimulus = stimulus.split(' ')
        center_of_fixation = round(len(stimulus[fixated_position_stimulus]) * 0.5)
        # find length of stimulus (in characters) up until fixated word
        len_till_fix = sum([len(token)+1 for token in stimulus[:fixated_position_stimulus]])
        eye_position = len_till_fix + center_of_fixation # + offset_from_word_center
    else:
        stim_indices, word_indices = [],[]
        for i, char in enumerate(stimulus + ' '):
            if char == ' ':
                stim_indices.append(word_indices)
                word_indices = []
            else:
                word_indices.append(i)
        eye_position = stim_indices[fixated_position_stimulus][eye_position]

    return int(np.round(eye_position))

def compute_ngram_activity(stimulus:str,
                           eye_position:int,
                           attention_position:int,
                           attend_width:float,
                           let_per_deg:float,
                           attention_skew:float,
                           gap:int,
                           recognition_in_stimulus:list[int],
                           tokens:list,
                           recognized_word_at_cycle:np.ndarray[int],
                           n_cycles:int)->dict:

    """
    Initialize word activity based on ngram excitatory input.

    :param stimulus: the tokens the model is processing in parallel.
    :param eye_position: the index of the character the eyes are fixating at in the stimulus.
    :param attention_position: the index of the character where the focus of attention is located in the stimulus.
    :param attend_width: how long the attention window should be when processing the input stimulus.
    :param let_per_deg: used to calculate visual acuity, which is then used to compute attention.
    :param attention_skew: used in the formula to compute attention. How skewed attention should be to the right of the fixation point. 1 equals symmetrical distribution.
    :param gap: the number of characters between two characters allowed to still form a bi-gram.
    :param recognition_in_stimulus: list of word indices that have been recognized in stimulus.
    :param tokens: the input text tokens.
    :param recognized_word_at_cycle: which processing cycle each word in the text has been recognized. -1 if word not yet recognized.
    :param n_cycles: how many processing cycles have already occurred in current fixation.

    :return: dict with ngram as keys and excitatory input as value.
    """

    unit_activations = {}
    # define the word ngrams, its weights and their location within the word
    all_ngrams, all_weights, all_locations = string_to_open_ngrams(stimulus, gap)
    fix_ngrams = []

    if len(recognition_in_stimulus) > 0 and len(tokens) > 0 and len(recognized_word_at_cycle) > 0 and n_cycles > -1:
        for i in recognition_in_stimulus:
            # AL: a hack to avoid recognized words to be too active and be matched to subsequent positions too!
            # after recognition, 200ms block on activation (= 8 act cycles)
            if n_cycles - recognized_word_at_cycle[i] <= 8:
                ngrams, weights, locations = string_to_open_ngrams(tokens[i], gap)
                fix_ngrams.extend(ngrams)
                # print(tokens[i], n_cycles, recognition_cycle[i], ngrams)

    for ngram, weight, location in zip(all_ngrams, all_weights, all_locations):
        # remove activation of ngrams from recognized words for the next 8 act cycles after recognition
        if ngram in fix_ngrams:
            activation = 0.0
        else:
            activation = cal_ngram_exc_input(location, weight, eye_position, attention_position,
                                             attend_width, let_per_deg, attention_skew)
        # AL: a ngram that appears more than once in the simulus
        # will have the activation from the ngram in the position with highest activation
        if ngram in unit_activations.keys():
            unit_activations[ngram] = max(unit_activations[ngram], activation)
        else:
            unit_activations[ngram] = activation
    # print(unit_activations)

    return unit_activations

def compute_words_input(stimulus:str,
                        lexicon_word_ngrams:dict,
                        eye_position:int,
                        attention_position:int,
                        attend_width:float,
                        pm,
                        freq_dict:dict,
                        recognition_in_stimulus:list[int],
                        tokens:list[str],
                        recognized_word_at_cycle:np.ndarray[int],
                        n_cycles:int)->(np.ndarray,list,int,int):

    """
    Calculate activity for each word in the lexicon given the excitatory input from all ngrams in the stimulus.

    :param stimulus: the tokens the model is processing in parallel.
    :param lexicon_word_ngrams: dict mapping words in the lexicon and the respective ngrams each word generates.
    :param eye_position: the index of the character the eyes are fixating at in the stimulus.
    :param attention_position: the index of the character where the focus of attention is located in the stimulus.
    :param attend_width: how long the attention window should be when processing the input stimulus.
    :param pm: the model attributes, set when ReadingModel is initialised.
    :param freq_dict: dict mapping words and its frequencies.
    :param recognition_in_stimulus: list of word indices that have been recognized in stimulus.
    :param tokens: the input text tokens.
    :param recognized_word_at_cycle: which processing cycle each word in the text has been recognized. -1 if word not yet recognized.
    :param n_cycles: how many processing cycles have already occurred in current fixation.

    :return: word_input (np.ndarray) with the resulting activity for each word in the lexicon,
    all_ngrams (list) with the number of ngrams per word in the lexicon,
    total_ngram_activity (int) as the total activity ngrams in the input resonate in the lexicon,
    n_ngrams (int) as the total number of ngrams in the input.
    """

    lexicon_size = len(lexicon_word_ngrams.keys())
    word_input = np.zeros(lexicon_size, dtype=float)

    # define ngram activity given stimulus
    unit_activations = compute_ngram_activity(stimulus, eye_position,
                                              attention_position, attend_width, pm.let_per_deg,
                                              pm.attention_skew, pm.ngram_gap,
                                              recognition_in_stimulus, tokens, recognized_word_at_cycle, n_cycles)
    # print(f'Activated ngrams: {unit_activations}')
    total_ngram_activity = sum(unit_activations.values())
    n_ngrams = len(unit_activations.keys())

    # compute word input according to ngram excitation and inhibition
    # all stimulus bigrams used, therefore the same bigram inhibition for each word of lexicon
    # (ngram excit is specific to word, ngram inhib same for all)
    ngram_inhibition_input = sum(unit_activations.values()) * pm.ngram_to_word_inhibition
    for lexicon_ix, lexicon_word in enumerate(lexicon_word_ngrams.keys()):
        word_excitation_input = 0
        # ngram (bigram & monogram) activations
        ngram_intersect_list = set(unit_activations.keys()).intersection(set(lexicon_word_ngrams[lexicon_word]))
        for ngram in ngram_intersect_list:
            word_excitation_input += pm.ngram_to_word_excitation * unit_activations[ngram]
        # change activation based on frequency
        if freq_dict and lexicon_word in freq_dict.keys():
            word_excitation_input = word_excitation_input * (freq_dict[lexicon_word]**pm.freq_weight) # / len(lexicon_word) * pm.len_weight
        word_input[lexicon_ix] = word_excitation_input + ngram_inhibition_input

    # normalize based on number of ngrams in lexicon
    # MM: Add discounted_Ngrams to nr ngrams. Decreases input to short words
    # to compensate for fact that higher prop of their bigrams have higher wgt because edges
    all_ngrams = [len(ngrams) for ngrams in lexicon_word_ngrams.values()]
    word_input = word_input / (np.array(all_ngrams) + pm.discounted_ngrams)

    return n_ngrams, total_ngram_activity, all_ngrams, word_input

def update_word_activity(lexicon_word_activity:np.ndarray[float],
                         word_inhibition_matrix:np.ndarray[float],
                         pm,
                         word_input:np.ndarray[float]):

    """
    In each processing cycle, re-compute word activity using word-to-word inhibition and decay.

    :param lexicon_word_activity: word activity for words in lexicon.
    :param word_inhibition_matrix: inhibition matrix for words in lexicon.
    :param pm: the model attributes, set when ReadingModel is initialised.
    :param word_input: word activity for words in lexicon, based on ngram excitatory input.

    :return: lexicon_word_activity (array) with updated activity for each word in the lexicon,
    lexicon_word_inhibition (array) with total inhibition for each word in the lexicon.
    """

    lexicon_size = len(lexicon_word_activity)
    # NV: the more active a certain word is, the more inhibition it will execute on its peers
    # Activity is multiplied by inhibition constant.
    # NV: then, this inhibition value is weighed by how much overlap there is between that word and every other.
    lexicon_normalized_word_inhibition = (100.0 / lexicon_size) * pm.word_inhibition
    lexicon_active_words = np.zeros(lexicon_size, dtype=bool)
    # find which words are active
    lexicon_active_words[(lexicon_word_activity > 0.0) | (word_input > 0.0)] = True
    overlap_select = word_inhibition_matrix[:, (lexicon_active_words == True)]
    lexicon_select = (lexicon_word_activity + word_input)[
                         (lexicon_active_words == True)] * lexicon_normalized_word_inhibition
    # This concentrates inhibition on the words that have most overlap and are most active
    lexicon_word_inhibition = np.dot((overlap_select ** 2), -(lexicon_select ** 2))
    # Combine word inhibition and input, and update word activity
    lexicon_total_input = np.add(lexicon_word_inhibition, word_input)

    # in case you want to set word-to-word inhibition off
    # lexicon_total_input = word_input
    # lexicon_word_inhibition = None

    # final computation of word activity
    # pm.decay has a neg value, that's why it's here added, not subtracted
    lexicon_word_activity_change = ((pm.max_activity - lexicon_word_activity) * lexicon_total_input) + \
                                   ((lexicon_word_activity - pm.min_activity) * pm.decay)
    lexicon_word_activity = np.add(lexicon_word_activity, lexicon_word_activity_change)
    # correct activity beyond minimum and maximum activity to min and max
    lexicon_word_activity[lexicon_word_activity < pm.min_activity] = pm.min_activity
    lexicon_word_activity[lexicon_word_activity > pm.max_activity] = pm.max_activity

    return lexicon_word_activity, lexicon_word_inhibition

def match_active_words_to_input_slots(order_match_check:list[int],
                                      stimulus:str,
                                      recognized_word_at_position:np.ndarray[str],
                                      lexicon_word_activity:np.ndarray[float],
                                      lexicon: list[str],
                                      min_activity: float,
                                      stimulus_position:list[int],
                                      word_length_similarity_constant:float,
                                      recognition_in_stimulus:list[int],
                                      lexicon_thresholds:np.ndarray[float],
                                      verbose:bool=True)->(np.ndarray[str], np.ndarray[float]):

    """
    Match active words to spatio-topic representation. Fill in the slots in the stimulus.
    The winner is the word with the highest activity above recognition threshold and of similar length.

    :param order_match_check: order in which words are matched to slots in stim
    :param stimulus: the tokens the model is processing in parallel.
    :param recognized_word_at_position: array storing which word received the highest activation in each text position.
    :param lexicon_word_activity: word activity for words in lexicon.
    :param lexicon: list of words in the lexicon.
    :param min_activity: minimum activity allowed for a word in the lexicon.
    :param stimulus_position: the position of each word in the stimulus in relation to the text.
    :param word_length_similarity_constant: how similar in length the lexicon word and the text word must be to be able to match.
    :param recognition_in_stimulus: list of word indices that have been recognized in stimulus.
    :param lexicon_thresholds: recognition threshold for each word in the lexicon.
    :param verbose: whether to show progress on shell.

    :return: recognized_word_at_position is the updated array of recognized words in each text position,
    lexicon_word_activity is the updated array with activity of each word in the lexicon
    """

    above_thresh_lexicon = np.where(lexicon_word_activity > lexicon_thresholds, 1, 0)
    # above_thresh_lexicon = np.where(lexicon_word_activity > max_threshold, 1, 0)

    for slot_to_check in range(len(order_match_check)):
        # slot_num is the slot in the stim (spot of still-unrecogn word) that we're checking
        slot_num = order_match_check[slot_to_check]
        word_index = slot_num
        # in continuous reading, recognized_word_at_position contains all words in text,
        # so word_index is the word position in the text (instead of in the stimulus)
        if stimulus_position:
            word_index = stimulus_position[slot_num]
        # if the slot has not yet been filled
        if not recognized_word_at_position[word_index]:
            # Check words that have the same length as word in the slot we're now looking for
            word_searched = stimulus.split()[slot_num]
            # MM: recognWrdsFittingLen_np: array with 1=wrd act above threshold, & approx same len
            # as to-be-recogn wrd (with 'word_length_similarity_constant' margin), 0=otherwise
            similar_length = np.array([int(is_similar_word_length(len(x.replace('_', '')),
                                                                  len(word_searched), word_length_similarity_constant)) for x in lexicon])
            recognized_words_fit_len = above_thresh_lexicon * similar_length
            # if at least one word matches (act above threshold and similar length)
            if int(np.sum(recognized_words_fit_len)):
                # Find the word with the highest activation in all words that have a similar length
                highest = np.argmax(recognized_words_fit_len * lexicon_word_activity)
                highest_word = lexicon[highest]
                recognition_in_stimulus.append(word_index)
                if verbose:
                    print(f'word in input: {word_searched}      recogn. winner highest act: {highest_word}')
                logger.info(f'word in input: {word_searched}      one w. highest act: {highest_word}')
                # The winner is matched to the slot,
                # and its activity is reset to minimum to not have it matched to other words
                recognized_word_at_position[word_index] = highest_word
                lexicon_word_activity[highest] = min_activity
                above_thresh_lexicon[highest] = 0

    return recognized_word_at_position, lexicon_word_activity, recognition_in_stimulus

def semantic_processing(tokens:list[str],
                        tokenizer,
                        language_model,
                        prediction_flag:str,
                        top_k:int|str = 'all',
                        threshold:float = 0.1,
                        device:torch.device = None)->dict:

    """
    Compute next-word predictions for each position in input text.

    :param tokens: words in input text.
    :param tokenizer: language model tokenizer from transformers library.
    :param language_model: the language model from transformers library to make predictions with.
    :param prediction_flag: which resource to use for predictions (a specific language model or cloze values from a cloze task)
    :param top_k: number of top predictions to use. If 'all', all predictions (above threshold) are returned.
    :param threshold: the minimum prediction score for word to be included.
    :param device: cpu or cuda, where to load the model.

    :return: dict with text position as key and predictions as values.
    """

    pred_info = dict()

    for i in range(1, len(tokens)):

        sequence = ' '.join(tokens[:i])
        # pre-process text
        encoded_input = tokenizer(sequence, return_tensors='pt')
        if device:
            encoded_input.to(device)
        # output contains at minimum the prediction scores of the language modelling head,
        # i.e. scores for each vocab token given by a feed-forward neural network
        output = language_model(**encoded_input)
        # logits are prediction scores of language modelling head;
        # of shape (batch_size, sequence_length, config.vocab_size)
        logits = output.logits[:, -1, :]
        # convert raw scores into probabilities (between 0 and 1)
        probabilities = nn.functional.softmax(logits, dim=1)

        # # add target word, also if subtoken
        target_word = tokens[i]
        if prediction_flag == 'gpt2':
            target_word = ' ' + tokens[i]
        target_token = tokenizer.encode(target_word, return_tensors='pt')

        if top_k == 'target_word':
            if target_token.size(dim=1) > 0:
                top_tokens = [target_word]
                target_id = target_token[0][0]
                # deals with quirk from llama of having <unk> as first token
                if prediction_flag == 'llama':
                    decoded_token = [tokenizer.decode(token) for token in target_token[0]]
                    if decoded_token[0] == '<unk>':
                        target_id = target_token[0][1]
                top_probabilities = [float(probabilities[0,target_id])]
                pred_info[i] = (top_tokens, top_probabilities)
        else:
            k = top_k
            if top_k == 'all':
                # top k is the number of probabilities above threshold
                if threshold:
                    above_threshold = torch.where(probabilities > threshold, True, False)
                    only_above_thrs = torch.masked_select(probabilities, above_threshold)
                    k = len(only_above_thrs)
                else:
                    k = len(probabilities[0])
            top_tokens = [tokenizer.decode(id.item()) for id in torch.topk(probabilities, k=k)[1][0]]
            top_probabilities = [float(pred) for pred in torch.topk(probabilities, k=k)[0][0]]
            # add target word if among top pred, also if subtoken
            target_tokens = [tokenizer.decode(token) for token in target_token[0]]
            target_tokens = [token for token in target_tokens if token != '<unk>']
            if target_tokens[0] in top_tokens:
                loc = top_tokens.index(target_tokens[0])
                top_tokens[loc] = target_word
            pred_info[i] = (top_tokens, top_probabilities)

    return pred_info

def activate_predicted_upcoming_word(position:int|str,
                                     target_word:str,
                                     fixation:int,
                                     lexicon_word_activity:np.ndarray[float],
                                     lexicon:list[str],
                                     pred_dict:dict,
                                     pred_weight:float,
                                     recognized_word_at_position:np.ndarray[str],
                                     pred_bool:bool,
                                     verbose:bool)->(np.ndarray, bool):
    """
    Activate predicted upcoming words for a given position.

    :param position: position of predicted upcoming word.
    :param target_word: the word at the position in the text.
    :param fixation: which token from input text the fixation is located at.
    :param lexicon_word_activity: word activity for words in lexicon.
    :param lexicon: list of words in the lexicon.
    :param pred_dict: word position as keys and predicted words with prediction scores as values.
    :param pred_weight: the weight of prediction score in changing word activation.
    :param recognized_word_at_position: array storing which word received the highest activation in each text position.
    :param pred_bool: whether predictive activation has occurred at this position.
    :param verbose: whether to print progress.

    :return: updated word activity in lexicon and whether predictive activation has occurred at this position.
    """

    if str(position) in pred_dict.keys():

        predicted = pred_dict[str(position)]

        if predicted['target'] != target_word and verbose:
            warnings.warn(f'Target word in predictability map "{predicted["target"]}" not the same as target word in model stimuli "{target_word}", position {position}')

        for token, pred in predicted['predictions'].items():

            if token in lexicon:
                i = lexicon.index(token)
                pred_previous_word = 0
                # determine the predictability of the previous text word to weight predictability of position
                if recognized_word_at_position[position - 1]:
                    pred_previous_word = 1
                # if previous word has not been recognized yet
                else:
                    # if position not the first word in the text and in predictability map
                    if position - 1 > 0 and str(position - 1) in pred_dict.keys():
                        # if previous text word is among the predictions
                        if pred_dict[str(position-1)]['target'] in pred_dict[str(position-1)]['predictions'].keys():
                            # and previous word to that word has been recognized
                            if position - 2 >= 0 and recognized_word_at_position[position - 2]:
                                # weight pred by the pred value of the previous word that is > 0 and < 1
                                pred_previous_word = pred_dict[str(position-1)]['predictions'][pred_dict[str(position-1)]['target']]
                                # pred_previous_word = entropy[position-1]

                # weight predictability with predictability (certainty) of previous text word
                if pred_previous_word:
                    # pre_act = (pred * pred_weight) / pred_previous_word
                    pre_act = (pred * pred_previous_word * pred_weight)
                    lexicon_word_activity[i] += pre_act

                    if position == fixation + 1 and pre_act > 0:
                        pred_bool = True

                    if verbose:
                        print(f'Word "{token}" received pre-activation <{round(pre_act,3)} ({pred} * {pred_previous_word} * {pred_weight})> in position of text word "{target_word}" ({round(lexicon_word_activity[i],3)} -> {round(lexicon_word_activity[i] + pre_act,3)})')
                    logger.info(f'Word "{token}" received pre-activation <{round(pre_act,3)} ({pred} * {pred_previous_word} * {pred_weight})> in position of text word "{target_word}" ({round(lexicon_word_activity[i],3)} -> {round(lexicon_word_activity[i] + pre_act,3)})')

                else:
                    logger.info(f'Word "{token} was not pre-activated ({pred} * {pred_previous_word} * {pred_weight}) in position of text word "{target_word}"')
                    if verbose:
                        print(f'Word "{token} was not pre-activated ({pred} * {pred_previous_word} * {pred_weight}) in position of text word "{target_word}"')
    else:
        if verbose:
            print(f'Position {position} not found in predictability map')
        logger.info(f'Position {position} not found in predictability map')

    return lexicon_word_activity, pred_bool

def compute_next_attention_position(reader_output:list,
                                    tokens:list[str],
                                    fixation:int,
                                    word_edges:dict[int,tuple[int,int]],
                                    fixated_position_in_stimulus:int,
                                    regression_flag:np.ndarray[bool],
                                    recognized_word_at_position:np.ndarray[str],
                                    lexicon_word_activity:np.ndarray[float],
                                    eye_position:int,
                                    fixation_counter:int,
                                    attention_position:int,
                                    fix_lexicon_index:int,
                                    pm,
                                    verbose:bool)->float:

    """
    Define where attention should be moved next based on recognition of words in current stimulus and the visual
    salience of the words to the right of fixation.

    :param reader_output: list of fixation outputs from model.
    :param tokens: list of tokens in text.
    :param fixation: which token from input text the fixation is located at.
    :param word_edges: dictionary storing word edges in stimulus.
    :param fixated_position_in_stimulus: the position of the fixated word in relation to the stimulus
    :param regression_flag: history of regressions, set to true at a certain position in the text when a regression is performed to that word.
    :param recognized_word_at_position: array storing which word received the highest activation in each text position.
    :param lexicon_word_activity: word activity for words in lexicon.
    :param eye_position: eye position in stimulus (in number of characters).
    :param fixation_counter: how many fixations have been performed by this point in processing the current text.
    :param attention_position: attention position in stimulus (in number of characters).
    :param fix_lexicon_index: index of fixated word in lexicon.
    :param pm: an instance of ReadingModel.
    :param verbose: whether to print progress.


    :return: the next attention position as the index of the letter in the word programmed to be fixated next (in relation to total stimulus characters).
    """

    # Define target of next fixation relative to fixated word n (i.e. 0=next fix on word n, -1=fix on n-1, etc). Default is 1 (= to word n+1)
    next_fixation = 1
    refix_size = pm.refix_size

    # regression: check whether previous word was recognized or there was already a regression performed. If not: regress
    if fixation > 0 and not recognized_word_at_position[fixation - 1] and not regression_flag[fixation - 1]:
        next_fixation = -1

    # skip bcs regression: if the current fixation was a regression
    # elif regression_flag[fixation]:
    #     # go to the nearest non-recognized word to the right within stimulus
    #     for i in [1, 2]:
    #         if fixation + i < len(tokens):
    #             if recognized_word_at_position[fixation + i]:
    #                 next_fixation = i + 1

    # refixation: refixate if the foveal word is not recognized but is still being processed
    elif (not recognized_word_at_position[fixation]) and (lexicon_word_activity[fix_lexicon_index] > 0):
        # # AL: only allows 3 consecutive refixations on the same word to avoid infinitely refixating if no word reaches threshold recognition at a given position
        # refixate = check_previous_refixations_at_position(all_data, fixation, fixation_counter, max_n_refix=3)
        # print(refixate)
        # if refixate:
        word_reminder_length = word_edges[fixated_position_in_stimulus][1] - eye_position
        if verbose:
            print('Refixating... Word reminder length: ', word_reminder_length)
        if word_reminder_length > 0:
            next_fixation = 0
            if fixation_counter - 1 in reader_output:
                if not reader_output[fixation_counter - 1].saccade_type == 'refixation':
                    refix_size = np.round(word_reminder_length * refix_size)
                    if verbose:
                        print('refix size: ', refix_size)

    # skip bcs next word has already been recognized
    elif fixation + 1 < len(tokens) and recognized_word_at_position[fixation + 1] and fixation + 2 < len(tokens):
        next_fixation = 2
        if recognized_word_at_position[fixation + 2] and fixation + 3 < len(tokens):
            next_fixation = 3

    # forward saccade: perform normal forward saccade (unless at the last position in the text)
    elif fixation < (len(tokens) - 1):
        word_attention_right = calc_word_attention_right(word_edges,
                                                         eye_position,
                                                         attention_position,
                                                         pm.attend_width,
                                                         pm.salience_position,
                                                         pm.attention_skew,
                                                         fixated_position_in_stimulus,
                                                         verbose)
        next_fixation = word_attention_right.index(max(word_attention_right))
    if verbose:
        print(f'next fixation: {next_fixation}')
    logger.info(f'next fixation: {next_fixation}')

    # AL: Calculate next attention position based on next fixation estimate = 0: refixate, 1: forward, 2: wordskip, -1: regression
    if next_fixation == 0:
        # MM: if we're refixating same word because it has highest attentwgt AL: or not being recognized whilst processed
        # ...use first refixation middle of remaining half as refixation stepsize
        fixation_first_position_right_to_eye = eye_position + 1 if eye_position + 1 < len(tokens) else eye_position
        attention_position = fixation_first_position_right_to_eye + refix_size

    elif next_fixation in [-1, 1, 2, 3]:
        attention_position = get_midword_position_for_surrounding_word(next_fixation, word_edges, fixated_position_in_stimulus)

    if verbose:
        print(f'attentpos {attention_position}')
    logger.info(f'attentpos {attention_position}')

    return attention_position

# def compute_next_attention_position(reader_output:list,
#                                     tokens:list[str],
#                                     fixation:int,
#                                     word_edges:dict[int,tuple[int,int]],
#                                     fixated_position_in_stimulus:int,
#                                     regression_flag:np.ndarray[bool],
#                                     recognized_word_at_position:np.ndarray[str],
#                                     lexicon_word_activity:np.ndarray[float],
#                                     eye_position:int,
#                                     fixation_counter:int,
#                                     attention_position:int,
#                                     fix_lexicon_index:int,
#                                     pm,
#                                     verbose:bool)->float:
#
#     """
#     Define where attention should be moved next based on recognition of words in current stimulus and the visual
#     salience of the words to the right of fixation.
#
#     :param reader_output: list of fixation outputs from model.
#     :param tokens: list of tokens in text.
#     :param fixation: which token from input text the fixation is located at.
#     :param word_edges: dictionary storing word edges in stimulus.
#     :param fixated_position_in_stimulus: the position of the fixated word in relation to the stimulus
#     :param regression_flag: history of regressions, set to true at a certain position in the text when a regression is performed to that word.
#     :param recognized_word_at_position: array storing which word received the highest activation in each text position.
#     :param lexicon_word_activity: word activity for words in lexicon.
#     :param eye_position: eye position in stimulus (in number of characters).
#     :param fixation_counter: how many fixations have been performed by this point in processing the current text.
#     :param attention_position: attention position in stimulus (in number of characters).
#     :param fix_lexicon_index: index of fixated word in lexicon.
#     :param pm: an instance of ReadingModel.
#     :param verbose: whether to print progress.
#
#
#     :return: the next attention position as the index of the letter in the word programmed to be fixated next (in relation to total stimulus characters).
#     """
#
#     # Define target of next fixation relative to fixated word n (i.e. 0=next fix on word n, -1=fix on n-1, etc). Default is 1 (= to word n+1)
#     next_fixation = 1
#
#
#
#     if verbose:
#         print(f'attentpos {attention_position}')
#     logger.info(f'attentpos {attention_position}')
#
#     return attention_position

def compute_next_eye_position(pm,
                              attention_position:int,
                              eye_position:int,
                              fixated_position_in_stimulus:int,
                              fixation:int,
                              word_edges:dict[int,tuple[int,int]],
                              verbose:bool)->(int, int, int, float, str):
    """
    This function computes next eye position using saccade distance
    (defined by next attention position and current eye position) plus a saccade error.

    :param pm: instance of ReadingModel.
    :param attention_position: attention position in stimulus (in number of characters).
    :param eye_position: eye position in stimulus (in number of characters).
    :param fixated_position_in_stimulus: the position of the fixated word in relation to the stimulus.
    :param fixation: which token from input text the fixation is located at.
    :param word_edges: dictionary storing word edges in stimulus.
    :param verbose: whether to print progress.

    :return: the next fixated word index, the next eye position (which character from fixated word) and saccade info for next fixation
    """

    # saccade distance is next attention position minus the current eye position
    saccade_distance = attention_position - eye_position
    if verbose:
        print(f'saccade distance: {saccade_distance}')
    logger.info(f'saccade distance: {saccade_distance}')

    # normal random error based on difference with optimal saccade distance
    saccade_error = calc_saccade_error(saccade_distance,
                                       pm.sacc_optimal_distance,
                                       pm.sacc_err_scaler,
                                       pm.sacc_err_sigma,
                                       pm.sacc_err_sigma_scaler,
                                       pm.use_saccade_error)


    saccade_distance = saccade_distance + saccade_error
    if verbose:
        print(f'saccade error: {saccade_error}')
    logger.info(f'saccade error: {saccade_error}')

    # offset_from_word_center = saccade_info['offset from word center'] + saccade_error
    saccade_distance = float(saccade_distance)
    saccade_error = float(saccade_error)

    # compute the position of next fixation
    # eye_position = int(np.round(eye_position + saccade_distance))
    if saccade_distance < 0:
        eye_position = int(math.floor(eye_position + saccade_distance))
    else:
        eye_position = int(math.ceil(eye_position + saccade_distance))
    if verbose:
        print(f'next eye position: {eye_position}')
    logger.info(f'next eye position: {eye_position}')

    # determine next fixation depending on next eye position
    fixation_saccade_map = {0: 'refixation',
                            -1: 'regression',
                            1: 'forward',
                            2: 'wordskip'}
    eye_pos_in_fix_word = None
    saccade_type = ''
    word_letter_indices = [i for edges in word_edges.values() for i in range(edges[0], edges[1]+1)]
    # if eye position is on a space, correct eye position to the closest letter to the right.
    if eye_position not in word_letter_indices:
        edges_indices = [edges[i] for edges in word_edges.values() for i in range(len(edges))]
        eye_position = min(edges_indices, key=lambda x: abs(x - eye_position)) + 2
    # find the next fixated word based on new eye position and determine saccade type based on that
    for word_i, edges in word_edges.items():
        if not eye_pos_in_fix_word:
            for letter_index, letter_index_in_stim in enumerate(range(edges[0], edges[1]+1)):
                if eye_position == letter_index_in_stim:
                    move = word_i - fixated_position_in_stimulus
                    fixation += move
                    # index of letter within fixated word to find eye position in relation to stimulus in the next fixation
                    eye_pos_in_fix_word = letter_index
                    if move > 2: move = 2
                    elif move < -1: move = -1
                    saccade_type = fixation_saccade_map[move]
                    break

    return fixation, eye_pos_in_fix_word, saccade_distance, saccade_error, saccade_type

class SequenceReader:
    def __init__(self,
                 model,
                 task,
                 text_id:int,
                 tokens:list[str]):

        """
        Store model, task and tokens from specific model.read() call.

        :param model: instance of ReadingModel.
        :param task: instance of TaskAttributes.
        :param text_id: id of text being read.
        :param tokens: input tokens from text to be read.
        """

        self.model = model
        self.task = task
        self.tokens = tokens
        self.text_id = text_id
        # save predictions from specific text being read
        self.text_predictions = None
        if model.predictability_values and str(text_id) in model.predictability_values.keys():
            self.text_predictions = model.predictability_values[str(text_id)]
        # save output from fixations
        self.output = []

class FixationOutput:
    def __init__(self,
                 reader:SequenceReader,
                 fixation:int=0,
                 stimulus:str='',
                 fixated_word_threshold:float=.0,
                 saccade_type:str='',
                 saccade_distance:int=0,
                 saccade_error:int=0,
                 saccade_cause:int=0,
                 recognition_cycle:int=-1,
                 fixation_duration:int=0,
                 recognized_words:np.ndarray=None,
                 recognized_word:str=''
                 ):

        """
        Store output from FixationProcessor.fixate().

        :param reader: an instance of SequenceReader.
        :param fixation: which token from input text the fixation is located at.
        :param stimulus: the tokens the model is processing in parallel.
        :param fixated_word_threshold: word recognition threshold of fixated word.
        :param saccade_type: the type of incoming saccade: skipping, forward, regression or refixation.
        :param saccade_distance: the incoming saccade distance. Distance between current eye position and eye previous position
        :param saccade_error: the error between the intended fixation point and where eyes actually land. Saccade noise to include overshooting.
        :param saccade_cause: for wordskip and refixation, extra info on cause of saccade.
        :param recognition_cycle: in which processing cycle the fixated word was recognized (if any). -1 if not recognized while being fixated.
        :param fixation_duration: the duration of the fixation.
        :param recognized_words: list of recognized words in text.
        :param recognized_word: which word was recognized in fixated position.
        """

        self.fixation = fixation
        self.fixated_word_threshold = fixated_word_threshold
        self.stimulus = stimulus
        self.saccade_type = saccade_type
        self.saccade_distance = saccade_distance
        self.saccade_error = saccade_error
        self.saccade_cause = saccade_cause
        self.recognition_cycle = recognition_cycle
        self.fixation_duration = fixation_duration
        self.recognized_word = recognized_word
        self.recognized_words = recognized_words
        self.fixated_word = reader.tokens[fixation]
        self.fixated_word_length = len(self.fixated_word)
        self.fixated_word_frequency = reader.model.frequency_values[reader.tokens[self.fixation]] if reader.model.frequency_values and reader.tokens[fixation] in reader.model.frequency_values.keys() else 0
        self.fixated_word_predictability = reader.text_predictions[str(fixation)][reader.tokens[fixation]] if reader.text_predictions and str(fixation) in reader.text_predictions.keys() and reader.tokens[self.fixation] in reader.text_predictions[str(fixation)].keys() else 0

    def to_dict(self):

        """
        Convert FixationOutput to dictionary.
        :return: dictionary with attributes as values.
        """

        return {
            'fixation': self.fixation,
            'fixated_word': self.fixated_word,
            'stimulus': self.stimulus,
            'fixated_word_length': self.fixated_word_length,
            'fixated_word_frequency': self.fixated_word_frequency,
            'fixated_word_predictability': self.fixated_word_predictability,
            'fixated_word_threshold': self.fixated_word_threshold,
            'saccade_type': self.saccade_type,
            'saccade_distance': self.saccade_distance,
            'saccade_error': self.saccade_error,
            'saccade_cause': self.saccade_cause,
            'recognition_cycle': self.recognition_cycle,
            'recognized_words': self.recognized_words,
            'recognized_word': self.recognized_word,
            'fixation_duration': self.fixation_duration,
        }

class FixationProcessor:

    def __init__(self,
                 reader:SequenceReader):

        """
        Process information from input text during reading.
        :param reader: an instance of SequenceReader.
        """

        self.reader = reader
        # AL: attributes initialized here change during processing and are needed for computing the output
        self.end_of_text = False  # set to true when end of text is reached
        self.eye_position = None  # initialize eye position
        self.stimulus = ''
        self.stimulus_position = None
        self.fixated_position_in_stimulus = None
        self.predicted = False  # keep track whether a prediction was made at a given position (for pred-att mechanism)
        self.fixation = 0  # the element of fixation in the text. It goes backwards in case of regression
        self.fixation_counter = 0  # +1 with every next fixation
        self.attention_position = None # initialize attention position
        self.attention_shift = False  # whether attention has shifted to another location (e.g. somewhere in the next word)
        self.n_cycles_since_attent_shift = 0  # number of cycles since attention shifted
        self.n_cycles = 0  # total number of cycles in the fixation
        self.recognition_in_stimulus = []  # stimulus position in which recognition is achieved during current fixation
        self.word_edges = dict() # the position of word edges in the stimulus
        self.lexicon_word_activity = np.zeros((len(reader.model.lexicon)), dtype=float)  # word activity for words in lexicon
        self.regression_flag = np.zeros(len(reader.tokens),
                                        dtype=bool)  # history of regressions, set to true at a certain position in the text when a regression is performed to that word
        self.recognized_word_at_position = np.empty(len(reader.tokens),
                                                    dtype=object)  # recognized word at position, which word received the highest activation in each position
        self.recognized_word_at_cycle = np.zeros(len(reader.tokens),
                                                 dtype=int)  # the amount of cycles needed for each word in text to be recognized
        self.recognized_word_at_cycle.fill(-1)  # initial state no words are recognized yet
        self.output = FixationOutput(reader) # initialize fixation output

    def define_stimulus_and_eye_pos(self, verbose=True):

        """
        Define stimulus and eye position of current fixation.

        :param verbose: whether to show progress.
        """

        tokens = self.reader.tokens
        # make sure that fixation does not go over the end of the text. Needed for continuous reading
        self.fixation = min(self.fixation, len(self.reader.tokens) - 1)
        self.stimulus, self.stimulus_position, self.fixated_position_in_stimulus = compute_stimulus(self.fixation, tokens, self.reader.model.stimulus_window)
        self.eye_position = compute_eye_position(self.stimulus, self.fixated_position_in_stimulus, self.eye_position)
        # define index of letters at the words edges. Used in processing cycle loop.
        self.word_edges = find_word_edges(self.stimulus)
        if verbose: print(f"Fixated word: {tokens[self.fixation]}\nStimulus: {self.stimulus}\nEye position: {self.eye_position}")
        logger.info(f"Fixated word: {tokens[self.fixation]}\nStimulus: {self.stimulus}\nEye position: {self.eye_position}")

    def update_attention_width(self):

        """
        Update attention width according to whether there was a regression in the last fixation.
        """

        # if this fixation location is a result of regression
        if self.output.saccade_type == 'regression':
            # set regression flag to know that a regression has been realized towards this position
            self.regression_flag[self.fixation] = True
            # narrow attention width by 2 letters in the case of regressions
            self.reader.model.attend_width = max(self.reader.model.attend_width - 1.0, self.reader.model.min_attend_width)
        else:
            # widen attention by 0.5 letters in forward saccades
            self.reader.model.attend_width = min(self.reader.model.attend_width + 0.5, self.reader.model.max_attend_width)

    def fixate(self, reader_output:list, verbose=True)-> FixationOutput:

        """
        Process words during fixation in input text.
        :param reader_output: list of outputs from FixationProcessor.fixate(), which is called recursively.
        :param verbose: whether to print progress.
        :return: an instance of FixationOutput.
        """

        # ---------------------- Define order of slot-matching and word edges  ---------------------
        # define order to match activated words to slots in the stimulus
        # NV: the order list should reset when stimulus changes or with the first stimulus
        order_match_check = define_slot_matching_order(len(self.stimulus.split()), self.fixated_position_in_stimulus,
                                                       self.reader.model.attend_width)

        # ---------------------- Start processing of stimulus ---------------------
        # define current attention position where eyes are fixating now
        self.attention_position = self.eye_position
        # re-initialize here because they need to refresh every fixation
        self.attention_shift = False
        self.n_cycles_since_attent_shift = 0
        self.n_cycles = 0
        self.recognition_in_stimulus = []
        fixated_word_index = self.reader.model.lexicon.index(self.reader.tokens[self.fixation])
        # initiate output of fixation
        self.output = FixationOutput(self.reader,
                                     stimulus=self.stimulus,
                                     fixation=self.fixation,
                                     fixated_word_threshold = self.reader.model.recognition_thresholds[fixated_word_index])

        # ---------------------- Define word excitatory input ---------------------
        n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(self.stimulus,
                                                                                     self.reader.model.lexicon_word_ngrams,
                                                                                     self.eye_position,
                                                                                     self.attention_position,
                                                                                     self.reader.model.attend_width,
                                                                                     self.reader.model,
                                                                                     self.reader.model.frequency_values,
                                                                                     self.recognition_in_stimulus,
                                                                                     self.reader.tokens,
                                                                                     self.recognized_word_at_cycle,
                                                                                     self.n_cycles)

        # Counter n_cycles_since_attent_shift is 0 until attention shift (saccade program initiation),
        # then starts counting to 5 (because a saccade program takes 5 cycles, or 125ms.)
        while self.n_cycles_since_attent_shift < 5:

            # AL: recompute word input if word gets recognized such that ngram activation for recognized word is removed.
            # AL: if attention position is None, it means in the previous cycle attention has shifted and the new attention position is outside the text (end of text).
            if self.recognition_in_stimulus and self.attention_position is not None:
            # and not self.attention_shift:
            # AL: it used to be that we would only remove activation of recognized words until attention shifted.
            # AL: but then we would get too many repetitions (not enough removal of activation; same word getting recognized in the subsequent position)

                n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(self.stimulus,
                                                                                             self.reader.model.lexicon_word_ngrams,
                                                                                             self.eye_position,
                                                                                             self.attention_position,
                                                                                             self.reader.model.attend_width,
                                                                                             self.reader.model,
                                                                                             self.reader.model.frequency_values,
                                                                                             self.recognition_in_stimulus,
                                                                                             self.reader.tokens,
                                                                                             self.recognized_word_at_cycle,
                                                                                             self.n_cycles)

            # ---------------------- Update word activity per cycle ---------------------
            # Update word act with word inhibition (input remains same, so does not have to be updated)
            self.lexicon_word_activity, lexicon_word_inhibition = update_word_activity(self.lexicon_word_activity,
                                                                                       self.reader.model.word_inhibitions,
                                                                                       self.reader.model, word_input)

            # check activity of fixated word in lexicon
            foveal_word_activity = self.lexicon_word_activity[fixated_word_index]
            if verbose:
                print(f'CYCLE {self.n_cycles}    activ @fix {round(foveal_word_activity, 3)} inhib  #@fix {round(lexicon_word_inhibition[fixated_word_index], 6)}')
            logger.info(f'CYCLE {self.n_cycles}    activ @fix {round(foveal_word_activity, 3)} inhib  #@fix {round(lexicon_word_inhibition[fixated_word_index], 6)}')

            # ---------------------- Match words in lexicon to slots in input ---------------------
            # word recognition, by checking matching active wrds to slots
            self.recognized_word_at_position, self.lexicon_word_activity, self.recognition_in_stimulus = \
                match_active_words_to_input_slots(order_match_check,
                                                  self.stimulus,
                                                  self.recognized_word_at_position,
                                                  self.lexicon_word_activity,
                                                  self.reader.model.lexicon,
                                                  self.reader.model.min_activity,
                                                  self.stimulus_position,
                                                  self.reader.model.word_length_similarity_constant,
                                                  self.recognition_in_stimulus,
                                                  self.reader.model.recognition_thresholds,
                                                  verbose=verbose)

            # ---------------------- Pre-activate predicted words in the stimulus ---------------------
            if self.fixation < len(self.reader.tokens)-1 and self.reader.model.predictability_values:
                # gradually pre-activate words in stimulus (pred weighted by pred of previous word)
                for position in range(self.fixation+1, self.fixation+len(self.stimulus.split(' '))):
                    if 0 < position < len(self.reader.tokens):
                        if not self.recognized_word_at_position[position]:
                            self.lexicon_word_activity, self.predicted = activate_predicted_upcoming_word(position,
                                                                                                self.reader.tokens[position],
                                                                                                self.fixation,
                                                                                                self.lexicon_word_activity,
                                                                                                self.reader.model.lexicon,
                                                                                                self.reader.text_predictions,
                                                                                                self.reader.model.pred_weight,
                                                                                                self.recognized_word_at_position,
                                                                                                self.predicted,
                                                                                                verbose)

            # ---------------------- Make saccade decisions ---------------------
            # word selection and attention shift
            if not self.attention_shift:
                # MM: on every cycle, take sample (called shift_start) out of normal distrib.
                # If cycle since fixstart > sample, make attentshift. This produces approx ex-gauss SRT
                if self.recognized_word_at_position[self.fixation]:
                    recognized_flag = True
                else:
                    recognized_flag = False
                shift_start = sample_from_norm_distribution(self.reader.model.mu, self.reader.model.sigma, self.reader.model.recognition_speeding,
                                                            recognized=recognized_flag)
                # shift attention (& plan saccade in 125 ms) if n_cycles is higher than random threshold shift_start
                if self.n_cycles >= shift_start:

                    self.attention_shift = True
                    # recompute attention position because attention has shifted
                    self.attention_position = compute_next_attention_position(reader_output,
                                                                         self.reader.tokens,
                                                                         self.fixation,
                                                                         self.word_edges,
                                                                         self.fixated_position_in_stimulus,
                                                                         self.regression_flag,
                                                                         self.recognized_word_at_position,
                                                                         self.lexicon_word_activity,
                                                                         self.eye_position,
                                                                         self.fixation_counter,
                                                                         self.attention_position,
                                                                         fixated_word_index,
                                                                         self.reader.model,
                                                                         verbose)

                    # AL: new attention position becomes None if it falls at a position outside the text, so do not compute new words input
                    # attention_position may be zero if it's in the first character of the stimulus
                    if self.attention_position is not None:
                        # AL: recompute word input, using ngram excitation and inhibition, because attentshift changes bigram input
                        n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(self.stimulus,
                                                                                                     self.reader.model.lexicon_word_ngrams,
                                                                                                     self.eye_position,
                                                                                                     self.attention_position,
                                                                                                     self.reader.model.attend_width,
                                                                                                     self.reader.model,
                                                                                                     self.reader.model.frequency_values,
                                                                                                     self.recognition_in_stimulus,
                                                                                                     self.reader.tokens,
                                                                                                     self.recognized_word_at_cycle,
                                                                                                     self.n_cycles)
                        self.attention_position = np.round(self.attention_position)

                        if verbose: print(
                            f"  input after attentshift: {round(word_input[self.reader.model.lexicon.index(self.reader.tokens[self.fixation])], 3)}")
                        logger.info(
                            f"  input after attentshift: {round(word_input[self.reader.model.lexicon.index(self.reader.tokens[self.fixation])], 3)}")

            if self.attention_shift:

                self.n_cycles_since_attent_shift += 1  # ...count cycles since attention shift

            for i in self.stimulus_position:
                if self.recognized_word_at_position[i] and self.recognized_word_at_cycle[i] == -1:
                    # MM: here the time to recognize the word gets stored
                    self.recognized_word_at_cycle[i] = self.n_cycles
                    if i == self.fixation:
                        self.output.recognition_cycle = self.recognized_word_at_cycle[self.fixation]

            self.n_cycles += 1

        # ------- Out of cycle loop. After last cycle, register fixation output for fixated word before eye move is made -------
        self.output.recognized_words=self.recognized_word_at_position
        self.output.fixation_duration=self.n_cycles * self.reader.model.cycle_size

        if verbose:
            print(f"Fixation duration: {self.output.fixation_duration} ms.")
        logger.info(f"Fixation duration: {self.output.fixation_duration} ms.")

        if self.recognized_word_at_position[self.fixation]:
            self.output.recognized_word = self.recognized_word_at_position[self.fixation]
            if verbose:
                if self.recognized_word_at_position[self.fixation] == self.reader.tokens[self.fixation]:
                    if verbose:
                        print("Correct word recognized at fixation!")
                    logger.info("Correct word recognized at fixation!")
                else:
                    if verbose:
                        print(f"Wrong word recognized at fixation! (Recognized: {self.recognized_word_at_position[self.fixation]})")
                    logger.info(f"Wrong word recognized at fixation! (Recognized: {self.recognized_word_at_position[self.fixation]})")
        else:
            self.output.recognized_word = ""
            if verbose:
                print("No word was recognized at fixation position")
                print(f"Word with highest activation: {self.reader.model.lexicon[np.argmax(self.lexicon_word_activity)]}")
            logger.info("No word was recognized at fixation position")
            logger.info(f"Word with highest activation: {self.reader.model.lexicon[np.argmax(self.lexicon_word_activity)]}")

        self.fixation_counter += 1

        if verbose:
            print(f'att pos right before computing next eye position: {self.attention_position}')
        logger.info(f'att pos right before computing next eye position: {self.attention_position}')

        return self.output

    def move_fixation(self, end_of_text:bool, verbose:bool=True)->bool:

        """
        Compute where to move the eyes next.

        :param end_of_text: whether fixation is at the end of the text.
        :param verbose: whether to print progress.
        :return: whether previous fixation should be the last because the text has ended.
        """

        # ------------ Compute next eye position and thus next fixation -----------------
        # next attention position is None if, when computed, it falls in position outside the text.
        # In this case, the end of reading is assumed to be reached.
        if self.attention_position is not None:

            (self.fixation,
             self.eye_position,
             self.output.saccade_distance,
             self.output.saccade_error,
             self.output.saccade_type) = compute_next_eye_position(self.reader.model,
                                                                   self.attention_position,
                                                                   self.eye_position,
                                                                   self.fixated_position_in_stimulus,
                                                                   self.fixation,
                                                                   self.word_edges,
                                                                   verbose)

            # AL: Update saccade cause for next fixation
            if self.output.saccade_type == 'wordskip':
                if self.regression_flag[self.fixation]:
                    self.output.saccade_cause = 2  # AL: bcs n resulted from regression and n + 1 has been recognized
                else:
                    self.output.saccade_cause = 1  # AL: bcs n + 2 has highest attwght (letter excitation)

            elif self.output.saccade_type == 'refixation':
                if not self.recognized_word_at_position[self.fixation]:
                    self.output.saccade_cause = 1  # AL: bcs fixated word has not been recognized
                else:
                    self.output.saccade_cause = 2  # AL: bcs right of fixated word has highest attwght (letter excitation)

            if self.output.saccade_type:
                saccade_symbols = {'forward': ">->->->->->->->->->->->-",
                                   'wordskip': ">>>>>>>>>>>>>>>>>>>>>>>>",
                                   'refixation': '------------------------',
                                   'regression': '<-<-<-<-<-<-<-<-<-<-<-<-'}
                if verbose:
                    print(saccade_symbols[self.output.saccade_type])
                logger.info(saccade_symbols[self.output.saccade_type])

        # Check if end of text is reached.
        # if next fixation is outside text (attention_position = None).
        if self.attention_position is None: # (self.fixation == len(self.reader.tokens) - 1 and self.output.saccade_type not in ['refixation','regression']) |
            end_of_text = True
            if verbose:
                print('End of text!')
            logger.info('End of text!')

        return end_of_text

def sequence_read(model,
                  task,
                  words:list[str],
                  text_id:int=-1,
                  verbose:bool=True):

    """
    Read text sequence (=words).

    :param model: an instance of ReadingModel.
    :param task: an instance of TaskAttributes.
    :param words: the input text sequence tokenized into a list of words.
    :param text_id: the id of the text sequence being read. Needed to retrieve predictions in specific text, if predictability is used.
    :param verbose: whether to print progress.

    :return: a list of FixationOutputs (one per fixation).
    """

    reader = SequenceReader(model, task, text_id, words)
    fixation_processor = FixationProcessor(reader)
    end_of_text = False

    while not end_of_text:

        if verbose:
            print(f'---Fixation {fixation_processor.fixation_counter} at position {fixation_processor.fixation}---')
        logger.info(f'---Fixation {fixation_processor.fixation_counter} at position {fixation_processor.fixation}---')

        fixation_processor.define_stimulus_and_eye_pos(verbose=verbose)
        fixation_processor.update_attention_width()
        fixation_processor.output = fixation_processor.fixate(reader.output, verbose=verbose)
        reader.output.append(fixation_processor.output)
        end_of_text = fixation_processor.move_fixation(end_of_text, verbose=verbose)

    return reader.output