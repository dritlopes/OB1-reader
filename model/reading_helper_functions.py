import numpy as np
import math
import re
import logging

logger = logging.getLogger(__name__)

def get_stimulus_edge_positions(stimulus:str)->list[int]:

    """
    Get position of letters next to spaces in the stimulus.

    :param stimulus: word(s) in model input.
    :return: which positions the letters at the edge of the words are located in relation to whole stimulus.
    """

    stimulus_word_edge_positions = []
    for letter_position in range(1,len(stimulus)-1):
        if stimulus[letter_position+1] == " " or stimulus[letter_position-1] == " ":
            stimulus_word_edge_positions.append(letter_position-1)

    return stimulus_word_edge_positions

# def string_to_open_ngrams(string:str, gap:int)->(list[str],list[float],list[list[int]]):
#
#     """
#     Determine the ngrams the word(s) in the model input activates, their activation weights and positions in the string.
#
#     :param string: word(s) in model input.
#     :param gap: the number of characters between two characters allowed to still form a bi-gram.
#     :return: the ngrams the word activates, the weights of ngram activation for each ngram, and where they are located in the word.
#     """
#
#     all_ngrams, all_weights, all_locations = [], [], []
#
#     # AL: make sure string contains one space before and after for correctly finding word edges
#     string = " " + string + " "
#     edge_locations = get_stimulus_edge_positions(string)
#     string = string.strip()
#
#     for position, letter in enumerate(string):
#         weight = 0.5
#         # AL: to avoid ngrams made of spaces
#         if letter != ' ':
#             if position in edge_locations:
#                 weight = 1.
#                 # AL: increases weigth of unigrams with no crowding (one-letter words, e.g. "a")
#                 if 0 < position < len(string) - 1:
#                     if string[position-1] == ' ' and string[position+1] == ' ':
#                         weight = 3
#                 # AL: include monogram if at word edge
#                 all_ngrams.append(letter)
#                 all_weights.append(weight)
#                 all_locations.append([position])
#
#             # AL: find bigrams
#             for i in range(1, gap+1):
#                 # AL: make sure second letter of bigram does not cross the stimulus string nor the word
#                 if position+i >= len(string) or string[position+i] == ' ':
#                     break
#                 # Check if second letter in bigram is edge
#                 if position+i in edge_locations:
#                     weight = weight * 2
#                 bigram = letter+string[position+i]
#                 all_ngrams.append(bigram)
#                 all_locations.append([position, position+i])
#                 all_weights.append(weight)
#
#     return all_ngrams, all_weights, all_locations

def string_to_ngrams(string, bigramFrame=None, gap=0):
    """
    Determine the ngrams the word(s) in the model input activates, their activation weights, and positions in the string.

    :param string: word(s) in model input.
    :param bigramFrame: DataFrame containing bigram frequencies (required for closed ngrams).
    :param gap: the number of characters between two characters allowed to still form a bi-gram.
                When gap=0, closed ngrams logic is used; otherwise, open ngrams logic is applied.
    :return: the ngrams the word activates, the weights of ngram activation for each ngram, and where they are located in the word.
    """
    all_ngrams, all_weights, all_locations = [], [], []

    if gap==0 and bigramFrame is None:
        raise ValueError("bigramFrame must be provided for closed ngrams (gap=0)")

    # Preprocess string
    if gap == 0:
        string = string.strip()
    else:
        string = " " + string + " "
        edge_locations = get_stimulus_edge_positions(string)
        string = string.strip()

    for position, letter in enumerate(string):
        if letter == ' ':
            continue

        # Determine weight for unigrams
        weight = 1.0 if gap == 0 else 0.5
        if gap > 0 and position in edge_locations:
            weight = 1.0
            if 0 < position < len(string) - 1 and string[position - 1] == ' ' and string[position + 1] == ' ':
                weight = 3.0

        # Add unigrams
        all_ngrams.append(letter)
        all_weights.append(weight)
        all_locations.append([position])

        # Add edge-specific bigrams for closed ngrams
        if gap == 0:
            if position == 0 or (position > 0 and string[position - 1] == " "):  # Start of word
                all_ngrams.append(' ' + letter)
                all_weights.append(weight)
                all_locations.append([position])
            if position == len(string) - 1 or (position < len(string) - 1 and string[position + 1] == " "):  # End of word
                all_ngrams.append(letter + ' ')
                all_weights.append(weight)
                all_locations.append([position])

        # Add bigrams
        for i in range(1, gap + 1):
            if position + i >= len(string) or string[position + i] == ' ':
                break

            bigram = letter + string[position + i]
            bigram_weight = weight

            if gap == 0:  # Closed ngrams
                if i > 1:
                    bigram_weight = 0.5
                if bigramFrame is not None and (bigramFrame["bigram"] == bigram).any():
                    bigrFreq = bigramFrame.loc[bigramFrame["bigram"] == bigram]["freq"].values[0]
                    if gap == 1 and (position == 0 or (position > 0 and string[position - 1] == " ")):
                        bigram_weight *= 2 * bigrFreq  # Extra weight for start of word
                    elif position == len(string) - 2 or (position < len(string) - 2 and string[position + 2] == " "):
                        bigram_weight *= 2 * bigrFreq  # Extra weight for end of word
                    else:
                        bigram_weight *= bigrFreq
                    all_ngrams.append(bigram)
                    all_weights.append(bigram_weight)
                    all_locations.append([position, position + i])
            else:  # Open ngrams
                if position + i in edge_locations:
                    bigram_weight *= 2
                all_ngrams.append(bigram)
                all_weights.append(bigram_weight)
                all_locations.append([position, position + i])

    return all_ngrams, all_weights, all_locations

def is_similar_word_length(len1:int, len2:int, len_sim_constant:float)->bool:

    """
    Define whether two words have similar length in order to match in spatio-topic representation.

    :param len1: length of candidate word in model's lexicon.
    :param len2: length of text word.
    :param len_sim_constant: how much similarity between the two word lengths there must be.

    :return: whether the two words have similar lengths.
    """

    is_similar = False
    # NV: difference of word length  must be within a percentage (defined by len_sim_constant) of the length of the longest word
    if abs(len1-len2) < (len_sim_constant * max(len1, len2)):
        is_similar = True

    return is_similar

def get_blank_screen_stimulus(blank_screen_type:str)->str:

    """
    Decide what type of stimulus to show if blank screen.

    :param blank_screen_type: type of blank screen.

    :return: stimulus to show if blank screen.
    """

    stimulus = None

    if blank_screen_type == 'blank':
        stimulus = ""

    elif blank_screen_type == 'hashgrid':
        stimulus = "#####"  # NV: overwrite stimulus with hash grid

    elif blank_screen_type == 'fixation cross':
        stimulus = "+"

    return stimulus

def define_slot_matching_order(n_words_in_stim:int, fixated_position_stimulus:int, attend_width:float)->list[int]:

    """
    Slot-matching mechanism. Determine order in which words are matched to slots in stimulus.
    Words are checked in the order of its attention weight. The closer to the fixation point, the more attention weight.

    :param n_words_in_stim: how many words are there in the stimulus.
    :param fixated_position_stimulus: the position of the fixated word in relation to the stimulus
    :param attend_width: how long the attention window should be when processing the input stimulus.

    :return: list with word indices of stimulus in the order which they should be matched.
    """

    # AL: made computation dependent on position of fixated word (so we are not assuming anymore that fixation is always at the center of the stimulus)
    positions = [+1,-1,+2,-2,+3,-3] # MM: no 0 because fix position gets added elsewhere
    # AL: number of words checked depend on attention width. The narrower the attention width the fewer words matched.
    n_words_to_match = min(n_words_in_stim, (math.floor(attend_width/3)*2+1))
    # AL: add fixated position to always be checked first
    order_match_check = [fixated_position_stimulus]
    for i, p in enumerate(positions):
        if i < n_words_to_match-1:
            next_pos = fixated_position_stimulus + p
            if 0 <= next_pos < n_words_in_stim:
                order_match_check.append(next_pos)
    #print('slots to fill:', n_words_to_match)

    return order_match_check

def sample_from_norm_distribution(mu:float, sigma:float, recog_speeding:float, recognized:bool)->int:

    """
    Sample from the normal distribution to determine whether attention should shift.

    :param mu: the mean (centre) of the distribution.
    :param sigma: the standard deviation of the distribution. Must be greater than 0.
    :param recog_speeding: improve chance of moving attention at each processing cycle as the fixated word is recognised.
    :param recognized: whether fixated word has been recognised.

    :return: starting from which processing cycle attention should shift.
    """

    if recognized:
        return int(np.round(np.random.normal(mu - recog_speeding, sigma, 1)))
    else:
        return int(np.round(np.random.normal(mu, sigma, 1)))

def find_word_edges(stimulus:str)->dict[int,(int,int)]:

    """
    Define index of letters at the words edges in relation to stimulus.

    :param stimulus: the tokens the model is processing in parallel (during one fixation).

    :return: dictionary storing word edges in stimulus.
    """
    # MM: word_edges is dict, with key is token position (from max -2 to +2, but eg. in fst fix can be 0 to +2).
    #    Entry per key is tuple w. two elements, the left & right edges, coded in letter position
    word_edges = dict()

    # AL: regex used to find indices of word edges
    p = re.compile(r'\b\w+\b', re.UNICODE)

    # Get word edges for all words starting with the word at fixation
    for i, m in enumerate(p.finditer(stimulus)):
        word_edges[i] = (m.start(),m.end()-1)

    return word_edges

def get_midword_position_for_surrounding_word(word_position:int, word_edges:dict[int,tuple[int,int]], fixated_position_in_stimulus:int)->int:

    """
    Determine where the center is located of words around a fixed word.

    :param word_position: the next fixation in relation to the fixated word (-1, +1, +2, +3).
    :param word_edges: dictionary storing word edges in stimulus.
    :param fixated_position_in_stimulus: the position of the fixed word in stimulus.

    :return: the index of the letter closest to the center of the word to be fixated next.
    """

    word_center_position = None
    word_position_in_stimulus = fixated_position_in_stimulus + word_position
    if word_position_in_stimulus in word_edges.keys():
        word_slice_length = word_edges[word_position_in_stimulus][1] - word_edges[word_position_in_stimulus][0] + 1
        # AL: math.ceil makes sure the center is rounded up to the nearest integer.
        word_center_position = word_edges[word_position_in_stimulus][0] + math.ceil(word_slice_length/2.0) - 1

    return word_center_position

def get_attention_skewed(attention_width:float, attention_eccentricity:float, attention_skew:float):

    """
    Compute how much attention is at a certain letter located in the stimulus,
    depending on how distant it is from the center of attention (attention eccentricity),
    how skewed attention should be (attention skew), and how many letters are processed in parallel (attention_window).

    :param attention_width: how long the attention window should be when processing the input stimulus.
    :param attention_eccentricity: how distant the ngram letter is from the center of attention.
    :param attention_skew: how skewed attention should be to the right of the fixation point. 1 equals symmetrical distribution.

    :return: how much attention is placed in a certain letter in the stimulus.
    """

    if attention_eccentricity < 0:
        # Attention left
        attention = 1.0 / attention_width * math.exp(-(pow(abs(attention_eccentricity), 2)) /
                                                     (2 * pow(attention_width / attention_skew, 2))) + 0.25
    else:
        # Attention right
        attention = 1.0 / attention_width * math.exp(-(pow(abs(attention_eccentricity), 2)) /
                                                     (2 * pow(attention_width, 2))) + 0.25
    return attention

def calc_acuity(eye_eccentricity:int, let_per_deg:float)->float:

    """
    Compute visual acuity of a certain letter in relation to the center of fixation in the stimulus.

    :param eye_eccentricity: distance in characters between letter position and center of fixation.
    :param let_per_deg: weight of such distance in defining acuity.

    :return: visual acuity of a certain letter in the stimulus.
    """
    # ; 35.55556 is to make acuity at 0 degs eq. to 1
    return (1/35.555556)/(0.018*(eye_eccentricity*let_per_deg+1/0.64))

def cal_ngram_exc_input(ngram_location:list[int], ngram_weight:float, eye_position:int, attention_position:int, attend_width:float, let_per_deg:float, attention_skew:float)->float:

    """
    Compute excitatory input from ngram to word activity based on attention eccentricity, attention skewness and visual acuity.

    :param ngram_location: the location of each letter forming the ngram in relation to stimulus.
    :param ngram_weight: the weight of the ngram input which depends on the ngram location (e.g. ngrams at the word edges exert higher activation)
    :param eye_position: the location of the center of fixation in stimulus.
    :param attention_position: the location of the center of attention in stimulus (which becomes different from eye_position when attention shifts before the eyes do).
    :param attend_width: how long the attention window should be when processing the input stimulus.
    :param let_per_deg: weight of distance between letter and center of fixation in defining acuity.
    :param attention_skew: how skewed attention should be to the right of the fixation point. 1 equals symmetrical distribution.

    :return: the excitatory input of ngram to word activity.
    """

    total_exc_input = 1

    # ngram activity depends on distance of ngram letters to the centre of attention and fixation, and left/right is skewed using negative/positve att_ecc
    for letter_position in ngram_location:
        attention_eccentricity = letter_position - attention_position
        eye_eccentricity = abs(letter_position - eye_position)
        attention = get_attention_skewed(attend_width, attention_eccentricity, attention_skew)
        visual_accuity = calc_acuity(eye_eccentricity, let_per_deg)
        exc_input = attention * visual_accuity
        total_exc_input = total_exc_input * exc_input

    # AL: if ngram contains more than one letter, total excitatory input is squared
    if len(ngram_location) > 1:
        total_exc_input = math.sqrt(total_exc_input)

    # AL: excitation is regulated by ngram location. Ngrams at the word edges have a higher excitatory input.
    total_exc_input = total_exc_input * ngram_weight

    return total_exc_input

def calc_monogram_attention_sum(position_start:int,
                                position_end:int,
                                eye_position:int,
                                attention_position:int,
                                attend_width:float,
                                attention_skew:float,
                                is_fixated_word:bool)->float:

    """
    Compute attention of word to the right of the fixated word.

    :param position_start: the index of the first letter of the word.
    :param position_end: the index of the last letter of the word.
    :param eye_position: the index of the center of fixation in stimulus.
    :param attention_position: the index of the center of attention in stimulus.
    :param attend_width: how long the attention window should be when processing the input stimulus.
    :param attention_skew: how skewed attention should be to the right of the fixation point. 1 equals symmetrical distribution.
    :param is_fixated_word: whether the word is being fixated.

    :return:
    """

    # this is only used to calculate where to move next when forward saccade
    # MM: turns out this can be seriously simplified: the weightmultiplier code can go, and the eccentricity effect.
    sum_attention_letters = 0

    # AL: make sure letters to the left of fixated word are not included
    if is_fixated_word:
        position_start = eye_position + 1

    for letter_location in range(position_start, position_end+1):
        #monogram_locations_weight_multiplier = 1  #was .5. Changed to 1. What happens when attentwgt not influenced by edges (which increases wgt of small words)
        #if foveal_word:
        #    if letter_location == position_end:
        #        monogram_locations_weight_multiplier = 1. # 2.
        #elif letter_location in [position_start, position_end]:
        #    monogram_locations_weight_multiplier = 1. # 2.

        # Monogram activity depends on distance of monogram letters to the centre of attention and fixation
        attention_eccentricity = letter_location - attention_position
        # eye_eccentricity = abs(letter_location - eye_position)
        # print(attention_eccentricity, eye_eccentricity)a
        attention = get_attention_skewed(attend_width, attention_eccentricity, attention_skew)
        #visual_acuity = 1 # calc_acuity(eye_eccentricity, let_per_deg)
        sum_attention_letters += attention #* visual_acuity) * monogram_locations_weight_multiplier
        #print(f'     letter within-word position: {letter_location}, '
        #      f'ecc: {attention_eccentricity}, '
        #      f'att-based input: {attention}, '
        #      f'visual acc {visual_acuity}, '
        #      f'visual input: {(attention * visual_acuity) * monogram_locations_weight_multiplier}')

    return sum_attention_letters

def calc_word_attention_right(word_edges:dict[int,tuple[int,int]],
                              eye_position:int,
                              attention_position:int,
                              attend_width:float,
                              salience_position:float,
                              attention_skew:float,
                              fixated_position_in_stimulus:int,
                              verbose:bool)->list[float]:

    """
    Compute word attention to the right of the fixed word, when forward saccade.

    :param word_edges: dictionary storing word edges in stimulus.
    :param eye_position: the position of the fixed letter in stimulus.
    :param attention_position: the position of the fixed letter in stimulus (which becomes different from eye_position when attention shifts before the eyes do).
    :param attend_width: how long the attention window should be when processing the input stimulus.
    :param salience_position: used to compute attention position by multiplying it with attention width.
    :param attention_skew: how skewed attention should be.
    :param fixated_position_in_stimulus: the position of the fixed word in stimulus.
    :param verbose: whether to print progress.

    :return: a list with attention to each word to the right of the fixated word.
    """

    # MM: calculate list of attention wgts for all words in stimulus to right of fix.
    word_attention_right = []
    attention_position += round(salience_position*attend_width)
    # AL: predictability modulation of next attention position. Turns out not a good idea.
    # if fixation+1 in highest_predictions.keys():
    #     attention_position += (1+highest_predictions[fixation+1]) * round(salience_position * attend_width)
    #     if verbose:
    #         print(f'Predictability regulating attention position... highest predictability value: {highest_predictions[-1]}')
    #     logger.info(f'Predictability regulating attention position... highest predictability value: {highest_predictions[-1]}')
    # else:
    #     attention_position += round(salience_position * attend_width)
    if verbose:
        print('Calculating visual input for next attention position...')
    logger.info('Calculating visual input for next attention position...')

    for i, edges in word_edges.items():
        if verbose:
            print(f'Word position: {i}')
            print(f'att width: {attend_width}, '
                  f'salience: {salience_position}, '
                  f'att position: {attention_position}, '
                  f'att skew: {attention_skew}, ')
        logger.info(f'Word position: {i}')
        logger.info(f'att width: {attend_width}, '
                  f'salience: {salience_position}, '
                  f'att position: {attention_position}, '
                  f'att skew: {attention_skew}, ')

        # if n or n + x (but not n - x), so only fixated word or words to the right
        if i >= fixated_position_in_stimulus:
            # print(i, edges)
            word_start_edge = edges[0]
            word_end_edge = edges[1]

            is_fixated_word = False
            if i == fixated_position_in_stimulus:
                is_fixated_word = True

            # if eye position at last letter (right edge) of fixated word
            if is_fixated_word and eye_position == word_end_edge:
                # set attention wghts for (nonexisting) right part of fixated word to 0
                crt_word_monogram_attention_sum = 0
            else:
                crt_word_monogram_attention_sum = calc_monogram_attention_sum(word_start_edge, word_end_edge, eye_position, attention_position, attend_width, attention_skew, is_fixated_word)
            # print('word position and visual salience: ',i,crt_word_monogram_attention_sum)
            word_attention_right.append(crt_word_monogram_attention_sum)
            # print(f'visual salience of {i} to the right of fixation: {crt_word_monogram_attention_sum}')
            if verbose:
                print(f'    word visual input: {crt_word_monogram_attention_sum}')
            logger.info(f'    word visual input: {crt_word_monogram_attention_sum}')

    return word_attention_right

def calc_saccade_error(saccade_distance, optimal_distance, sacc_err_scaler, sacc_err_sigma, sacc_err_sigma_scaler, use_saccade_error:bool)->float:

    """
    Compute error when moving eyes to account for overshooting.

    :param saccade_distance: distance in letters between eye position and next attention position.
    :param optimal_distance: reference used to compute error when moving eyes forward.
    :param sacc_err_scaler: weight of saccade error.
    :param sacc_err_sigma: standard deviation when sampling from normal distribution.
    :param sacc_err_sigma_scaler: weight of saccade sigma.
    :param use_saccade_error: whether to use saccade error.

    :return: saccade error; normal random error based on difference with optimal saccade distance to account for overshooting.
    """

    saccade_error_norm = 0.

    if use_saccade_error:

        saccade_error = (optimal_distance - abs(saccade_distance)) * sacc_err_scaler
        saccade_error_sigma = sacc_err_sigma + (abs(saccade_distance) * sacc_err_sigma_scaler)
        saccade_error_norm = np.random.normal(saccade_error, saccade_error_sigma, 1)

    return saccade_error_norm