import time
import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
import string
from datetime import datetime
from itertools import combinations
from model_components import read_sequence, run_wordrecog
from reading_helper_functions import string_to_ngrams
import task_attributes
# print('imported to task attrib')
from utils import get_ngram_frequency_from_file, get_word_freq, pre_process_string, write_out_simulation_data, get_word_pred, return_predicted_tokens
# print('imported all')

# will create a new file everytime, stamped with date and time
now = datetime.now()
dt_string = now.strftime("_%m_%d_%H-%M-%S")
filename = f'logs/logfile{dt_string}.log'
if not os.path.isdir('logs'): os.mkdir('logs')
logging.basicConfig(filename=filename,
                    force=True,
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def add_lexicon(words:list,
                lexicon_filepath:str,
                frequency_values:dict,
                predictability_values:dict,
                include_predicted_without_frequencies:bool,
                save:bool) -> list:

    """ Creates the lexicon of the model, which is a list holding all words that will be processed in the model.
    The lexicon can be made out of the words in the task stimuli, the most frequent words according
    to a frequency resource, and/or the predicted words according to a predictability resource.

    :return: lexicon: contains all words to be processed by the model. The model's vocabulary."""

    logger.info('Generating lexicon...')

    if os.path.exists(lexicon_filepath):
        with open(lexicon_filepath, 'rb') as infile:
            lexicon = pickle.load(infile)
        logger.info('Lexicon loaded from memory.')

    else:

        if not words:
            words = []
            logger.warning('No text words given to lexicon.')

        lexicon = set(words)

        logger.info(f'Lexicon size including input words: {len(lexicon)}')

        if not frequency_values:
            frequency_values = {}
            logger.warning('No frequency resource given to lexicon.')

        lexicon = lexicon | set(frequency_values.keys())
        logger.info(f'Lexicon size including words from frequency resource: {len(lexicon)}')

        if not predictability_values:
            predictability_values = {}
            logger.warning('No predictability resource given to lexicon.')

        if include_predicted_without_frequencies:
            predicted_tokens = return_predicted_tokens(predictability_values)
            lexicon = lexicon | set(predicted_tokens)
            logger.info(f'Lexicon size including input words from predictability resource (which are not in frequency resource): {len(lexicon)}')

        if not lexicon:
            raise AttributeError('Lexicon is empty. Please provide words to populate lexicon.')

        lexicon = list(lexicon)

        # make sure no empty strings nor punctuations are part of the lexicon
        lexicon = [word for word in lexicon if word not in string.punctuation and word != "" and word != " "]
        logger.info(f'Lexicon size after cleaning of empty strings and punctuations: {len(lexicon)}')

        if save:
            dir_path = os.path.dirname(lexicon_filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(lexicon_filepath, 'wb') as outfile:
                pickle.dump(lexicon, outfile)
            if save:
                logger.info(f"Lexicon saved to {lexicon_filepath}")

    return lexicon

def add_lexicon_ngram_mapping(lexicon:list, bigram_frame, ngram_gap:int) -> dict:

    """
    Creates mapping between each word in the lexicon and its ngrams.
    :param lexicon: all words to be processed by the model. The model's vocabulary.
    :param ngram_gap: how many in btw letters still lead to bi-gram.
    :param bigram_frame: DataFrame containing bigram frequencies (required for closed ngrams).
    :return: lexicon_word_ngrams: contains ngrams each word in the lexicon generates.
    """

    if lexicon:
        # AL: mapping between words and ngrams in the lexicon. Which ngrams form each word in the lexicon?
        logger.info('Finding ngrams from lexicon...')

        lexicon_word_ngrams = dict()
        for i, word in enumerate(lexicon):
            # AL: weights and locations are not used for lexicon,
            # only the ngrams of the words in the lexicon for comparing them later with the ngrams activated in stimulus.
            all_word_ngrams, weights, locations = string_to_ngrams(string=word, bigram_frame=bigram_frame, gap=ngram_gap)
            lexicon_word_ngrams[word] = all_word_ngrams

        lexicon_word_ngrams = lexicon_word_ngrams
        logger.info('Lexicon word ngrams created.')
    else:
        raise AssertionError(
            'Lexicon is empty. Create model lexicon first.')

    return lexicon_word_ngrams

def add_recognition_thresholds(lexicon:list, max_threshold:float, max_activity:float, freq_weight:float,
                               frequency_values: dict,
                               use_threshold: bool) -> np.ndarray:

    """If use_threshold=true, each word in lexicon gets own thresholds for recognition as in Snell et al. (2018) OB1 version.
    If use_threshold=false or  no model lexicon is present, thresholds equal maximum word threshold set in the model initialization.

    :param use_threshold: whether to use thresholds as defined by word frequency. If not, set all thresholds to max_threshold.
    :param lexicon: contains all words to be processed by the model. The model's vocabulary.
    :param max_threshold: the maximum recognition threshold a word is allowed to have.
    :param max_activity: maximum activity a word in the lexicon is allowed to have.
    :param freq_weight: weight frequency has in defining word recognition threshold
    :param frequency_values: contains a frequency value for each word.
    :return lexicon_thresholds: recognition threshold for each word in the lexicon."""

    logger.info('\nSetting word recognition thresholds...')

    if lexicon:
        lexicon_thresholds = np.zeros((len(lexicon)), dtype=float)
        for i, word in enumerate(lexicon):
            # should always ensure that the maximum possible value of the threshold doesn't exceed the maximum allowable word activity
            if max_threshold > max_activity: max_threshold = max_activity
            word_threshold = max_threshold
            # if frequency values are provided, determine threshold accordingly
            if frequency_values and use_threshold:
                max_frequency = max(frequency_values.values())
                if word in frequency_values.keys():
                    word_frequency = frequency_values[word]
                    # let threshold be fun of word freq. freq_p weighs how strongly freq is
                    # (1=max, then thresh. 0 for most freq. word; <1 means less heavy weighting)
                    # from 0-1, inverse of frequency, scaled to 0(highest freq)-1(lowest freq)
                    word_threshold = max_threshold * (
                            (max_threshold / freq_weight) - word_frequency) / (
                                             max_frequency / freq_weight)
                    # AL: changed this to be like in the original paper
                    # word_threshold = 0.22 * ((max_frequency/freq_p) - word_frequency) / (max_frequency/freq_p)
                    # AL: alter the size of effect of frequency and pred as a function of length effect on threshold, as in paper
                    # word_threshold = word_threshold * (1 - .61**(-0.44*len(word)))
            lexicon_thresholds[i] = word_threshold

        logger.info(f'Successfully set word recognition thresholds.')

        return lexicon_thresholds

    else:
        raise AttributeError(
            "Lexicon is empty. Create model lexicon by calling add_lexicon() first or provide it as argument when initializing the model.")

def add_word_inhibition_matrix(lexicon:list, lexicon_word_ngrams:dict, matrix_filepath:str, matrix_parameters_filepath:str, save:bool) -> np.ndarray:

    """ Word inhibition matrix of dimensions (lexicon length, lexicon length) determines
    the inhibition activity words have on each other within the lexicon. Only words of similar
    length inhibit each other to the extent of their ngram overlap. A model lexicon is needed
    to create inhibition matrix.

    :param lexicon: all words to be processed by the model. The model's vocabulary.
    :param lexicon_word_ngrams: ngrams each word in the lexicon generates.
    :param matrix_filepath: path of matrix file.
    :param matrix_parameters_filepath: path of matrix parameter file.
    :param save: save matrix to file.
    :return: word_inhibition_matrix: matrix containing word-to-word inhibition value per pairwise word combination."""

    logger.info('\nComputing word-to-word inhibition matrix...')

    # NV: compare the previous params with the actual ones.
    previous_matrix_usable = False

    if os.path.exists(matrix_filepath):

        if os.path.exists(matrix_parameters_filepath):

            with open(matrix_parameters_filepath, "rb") as f:
                parameters_previous = pickle.load(f)

            size_of_file = os.path.getsize(matrix_filepath)

            # NV: The file size is also added as a check
            # the idea is that the matrix is fully dependent on these parameters alone.
            # So, if the parameters are the same, the matrix should be the same.
            if str(lexicon_word_ngrams) + str(len(lexicon)) + str(size_of_file) \
                    == parameters_previous:
                previous_matrix_usable = True

        if previous_matrix_usable:
            logger.info('Loading previously saved inhibition matrix...')
            with open(matrix_filepath, "rb") as f:
                word_inhibition_matrix = pickle.load(f)
                return word_inhibition_matrix

    elif not previous_matrix_usable:
        logger.info('No previous inhibition matrix. Creating one...')

        if lexicon and lexicon_word_ngrams:

            lexicon_size = len(lexicon)
            word_inhibition_matrix = np.zeros((lexicon_size, lexicon_size), dtype=float)

            # for word_1_index in range(lexicon_size):  # MM: receiving unit...
            #     # AL: make sure word1-word2, but not word2-word1 or word1-word1.
            #     for word_2_index in range(word_1_index + 1, lexicon_size):  # MM: sending unit
            #         word1, word2 = lexicon[word_1_index], lexicon[word_2_index]
            for pair in combinations(lexicon, 2):
                word1, word2 = pair[0], pair[1]
                # the degree of length similarity
                length_sim = 1 - (abs(len(word1) - len(word2)) / max(len(word1), len(word2)))
                # if not is_similar_word_length(len(word1), len(word2), pm.word_length_similarity_constant):
                #     continue
                # else:
                ngram_common = list(
                    set(lexicon_word_ngrams[word1]).intersection(set(lexicon_word_ngrams[word2])))
                n_total_overlap = len(ngram_common)
                # MM: now inhib set as proportion of overlapping bigrams (instead of nr overlap);
                word_1_index, word_2_index = lexicon.index(word1), lexicon.index(word2)
                #print("word1 ", word1, "word2 ", word2, "length sim", length_sim, "overlap ", n_total_overlap,
                #       " ngram overlap ", ngram_common)
                word_inhibition_matrix[word_1_index, word_2_index] = (n_total_overlap / (
                    len(lexicon_word_ngrams[word1]))) * length_sim
                word_inhibition_matrix[word_2_index, word_1_index] = (n_total_overlap / (
                    len(lexicon_word_ngrams[word2]))) * length_sim
                # print("inhib one way", word_overlap_matrix[word_1_index, word_2_index])
            logger.info('Successfully created inhibition matrix.')

            if save:
                if not matrix_filepath or not matrix_parameters_filepath:
                    matrix_filepath = '../data/processed/inhibition_matrix_previous.pkl'
                    matrix_parameters_filepath = '../data/processed/inhibition_matrix_parameters_previous.pkl'
                    os.makedirs(os.path.dirname(matrix_filepath), exist_ok=True)
                    os.makedirs(os.path.dirname(matrix_parameters_filepath), exist_ok=True)

                with open(matrix_filepath, "wb") as f:
                    pickle.dump(word_inhibition_matrix, f)

                size_of_file = os.path.getsize(matrix_filepath)
                with open(matrix_parameters_filepath, "wb") as f:
                    pickle.dump(str(lexicon_word_ngrams) + str(lexicon_size) + str(size_of_file), f)
                logger.info(f'Successfully loaded word-to-word inhibition matrix.')
                if save:
                    logger.info(f'Matrix saved to {matrix_filepath}')

            return word_inhibition_matrix

        elif not lexicon:
            raise AttributeError(
                "Lexicon is empty. Create model lexicon by calling add_lexicon() first or provide it as argument when initializing the model.")

        elif not lexicon_word_ngrams:
            raise AttributeError(
                "Lexicon-ngram mapping is empty. Create mapping by calling add_lexicon_ngram_mapping() first.")

def load_frames(frame_filepath:str|None=None):

    logger.info('\nAttempting to load frames...')

    if frame_filepath:

        if frame_filepath.endswith('.json'):

            with open(frame_filepath, "r") as f:
                frames = json.load(f)

            frames =  frames.get("responses")

            if frames:
                logger.info('Successfully loaded frames.')
                return frames

            else:
                logger.warning('No frames found. Make sure that the file is in the expected format (higher-level key should be "responses").')
                return None

        else:
            logger.warning('Frame filepath given not in json format. Make sure that the file is in the expected format.')
            return None

    else:
        logger.warning('No filepath to frames given. Frame evoking module is off.')
        return None

class ReadingModel:

    """Class object that holds all model settings and initializes the model when called."""

    def __init__(self,
                 texts:list[str]=None,
                 cycle_size:int = 25,
                 ngram_to_word_excitation:float = 1.0,
                 ngram_to_word_inhibition:float = 0.0,
                 word_inhibition:float = -1.3, # -2.0, -1.5, -1.3
                 min_activity:float = 0.0,
                 max_activity:float = 1.0,
                 decay:float = -0.06, # -0.02, -0.06, -0.08
                 discounted_ngrams:int = 5,
                 ngram_gap:int = 0,
                 max_threshold:float = 0.5,
                 use_threshold:bool = False,
                 freq_weight:float = 0.08,
                 word_length_similarity_constant:float = 0.15,
                 top_k:str|int = 'all',
                 pred_threshold:float = 0.01,
                 pred_weight:float = 0.1,
                 pred_source:str = 'gpt2',
                 attend_width:float = 5.0,
                 max_attend_width:float = 5.0,
                 min_attend_width:float = 3.0,
                 attention_skew:int = 3,
                 let_per_deg:float = .3,
                 refix_size:float = 0.2,
                 salience_position:float = 0.5,
                 sacc_optimal_distance:int = 7,
                 sacc_err_scaler:float = 0.2,
                 sacc_err_sigma:float = 0.17,
                 sacc_err_sigma_scaler:float = 0.06,
                 mu:int = 12,
                 sigma:int = 4,
                 use_saccade_error:bool = True,
                 recognition_speeding:float = 5.0,
                 use_frequency:bool = True,
                 use_predictability:bool = False,
                 language:str = 'english',
                 frequency_values: dict = None,
                 predictability_values: dict = None,
                 frequency_filepath:str = '',
                 predictability_filepath:str = '',
                 lexicon_filepath: str = '',
                 matrix_filepath: str = '',
                 matrix_parameters_filepath: str = '',
                 include_predicted_without_frequencies: bool = False,
                 save_lexicon: bool = False,
                 save_word_inhibition: bool = False,
                 frame_filepath: str = '',
                 verbose: bool = True
                 ):

        """
        Create instance of model.

        :param texts: list of texts to be used to generate the lexicon (and to be included in the frequency resource and predictability resource).
        If not given, the model attempts to generate lexicon with words from word frequency resource and/or word predictability resource.
        :param cycle_size: milliseconds that one model cycle is supposed to last (brain time, not model time).
        :param ngram_to_word_excitation: weight on ngram activation.
        :param ngram_to_word_inhibition: weight on ngram inhibition.
        :param word_inhibition: weight on inhibition word exert on its recognition competitors.
        :param min_activity: minimum activity a word in the lexicon is allowed to have.
        :param max_activity: maximum activity a word in the lexicon is allowed to have.
        :param decay: decay in word activation over time.
        :param discounted_ngrams: # MM: Max extra wgt bigrams do to edges in 4-letter wrd w. gap 3. Added to bi-gram count in compute_input formula to compensate.
        :param ngram_gap: how many in btw letters still lead to bi-gram.
        :param max_threshold: the maximum recognition threshold a word is allowed to have. # MM: changed because max activity changed from 1.3 to 1.
        :param use_threshold: whether to use recognition threshold modulated by word frequency. If not, set all thresholds to max_threshold.
        :param freq_weight: weight frequency has in defining word recognition threshold # MM: words not in corpus have no freq, repaired by making freq less important.
        :param word_length_similarity_constant: determines how similar the length of 2 words must be for them to be recognised as 'similar word length.
        :param top_k: in case of language model providing predictions, save only the k highest predictions.
        :param pred_threshold: in case of language model providing predictions, save only the predictions above certain threshold.
        :param pred_weight: scaling parameter in pre-activation formula.
        :param pred_source: the source which generates predictability values. Default is gpt2 language model. Supported options: gpt2, llama, or cloze. For cloze values, make sure you have the required file (see function create_pred_file in utils.py)
        :param attend_width: how long the attention window should be when processing the input stimulus.
        :param max_attend_width: maximum attention width; used in reading simulation where attend_with is dynamic.
        :param min_attend_width: minimum attention width; used in reading simulation where attend_with is dynamic.
        :param attention_skew: used in the formula to compute attention. How skewed attention should be to the right of the fixation point. 1 equals symmetrical distribution.
        :param let_per_deg: weight of eccentricity in defining visual acuity.
        :param refix_size: during refix, how many letters do we jump forward in the word?
        :param salience_position: used to compute attention position by multiplying it with attention width.
        :param sacc_optimal_distance: reference used to compute error when moving eyes forward.
        :param sacc_err_scaler: used to compute error when moving the eyes forward.
        :param sacc_err_sigma: basic sigma; used to compute error when moving the eyes forward.
        :param sacc_err_sigma_scaler: effect of distance on sigma; used to compute error when moving the eyes forward.
        :param mu: used to sample from normal distribution chance of moving attention at each processing cycle. The mean (centre) of the distribution/
        :param sigma: used to sample from normal distribution chance of moving attention at each processing cycle. The standard deviation of the distribution.
        :param recognition_speeding: used to improve chance of moving attention at each processing cycle as the fixated word is recognised.
        :param use_frequency: whether to use word frequency in model.
        :param use_predictability: whether to use word predictability in model.
        :param language: language of input texts.
        :param frequency_filepath: filepath to save word frequencies.
        :param predictability_filepath: filepath to save word predictability.
        :param frequency_values: resource with word frequencies. If not provided, and use_frequency is True, the model will try to generate word frequencies from default resources.
        :param predictability_values: resource with predictability values. If not provided, and use_predictability is True, the model will try to generate predictability values from default resources.
        :param lexicon_filepath: path to lexicon file.
        :param matrix_filepath: path to word inhibition matrix file.
        :param include_predicted_without_frequencies: whether to include predicted words in lexicon which are not in the frequency resource.
        :param save_lexicon: whether to write out model lexicon.
        :param save_word_inhibition: whether to write out model word inhibition matrix.
        :param frame_filepath: path to file with frames parsed from the input text using frame-RAG. Needed to switch on OB1-frame. See https://github.com/dritlopes/text2frames for the expected file format.
        :param verbose: whether to print progress on the terminal.
        """

        self.texts = texts
        self.cycle_size = cycle_size
        self.ngram_to_word_excitation = ngram_to_word_excitation
        self.ngram_to_word_inhibition = ngram_to_word_inhibition
        self.word_inhibition = word_inhibition
        self.min_activity = min_activity
        self.max_activity = max_activity
        self.decay = decay
        self.discounted_ngrams = discounted_ngrams
        self.ngram_gap = ngram_gap
        self.max_threshold = max_threshold
        self.use_threshold = use_threshold
        self.freq_weight = freq_weight
        self.word_length_similarity_constant = word_length_similarity_constant
        self.top_k = top_k
        self.pred_source = pred_source
        self.pred_threshold = pred_threshold
        self.pred_weight = pred_weight
        self.attend_width = attend_width
        self.max_attend_width = max_attend_width
        self.min_attend_width = min_attend_width
        self.attention_skew = attention_skew
        self.let_per_deg = let_per_deg
        self.refix_size = refix_size
        self.salience_position = salience_position
        self.sacc_optimal_distance = sacc_optimal_distance
        self.sacc_err_scaler = sacc_err_scaler
        self.sacc_err_sigma = sacc_err_sigma
        self.sacc_err_sigma_scaler = sacc_err_sigma_scaler
        self.mu = mu
        self.sigma = sigma
        self.use_saccade_error = use_saccade_error
        self.recognition_speeding = recognition_speeding
        self.language = language
        self.frequency_filepath = frequency_filepath
        self.predictability_filepath = predictability_filepath
        self.lexicon_filepath = lexicon_filepath
        self.matrix_filepath = matrix_filepath
        self.matrix_parameters_filepath = matrix_parameters_filepath
        self.include_predicted_without_frequencies = include_predicted_without_frequencies
        self.save_lexicon = save_lexicon
        self.save_word_inhibition = save_word_inhibition
        self.verbose = verbose

        self.time = dt_string
        start_time = time.perf_counter()

        self.tokens = [pre_process_string(token) for text in texts for token in text.split(' ') if pre_process_string(token) != '']

        # make sure data directory is created if non-existent
        os.makedirs('../data/raw', exist_ok=True)
        os.makedirs('../data/processed', exist_ok=True)

        # if ngram gap is 0, load ngram frequencies
        if self.ngram_gap == 0:
            self.bigram_frame = get_ngram_frequency_from_file(language)
        else:
            self.bigram_frame = None

        # if use predictability but no predictability dict is provided, create predictability dict (or loaded it in if already exists by giving the filepath) if texts are provided and if language and pred_source are supported
        if texts and use_predictability and not predictability_values:
            predictability_values = get_word_pred(texts, language=language,
                                                  pred_source=pred_source,
                                                  pred_threshold=pred_threshold,
                                                  topk=top_k,
                                                  output_word_pred_map=predictability_filepath)
        self.predictability_values = predictability_values

        # if use frequency but not frequency dict is provided, create frequency dict (or load it if it already exists by giving the filepath) with default resources if language is supported
        if use_frequency and not frequency_values:
            frequency_values = get_word_freq(words=self.tokens,
                                             language=language,
                                             word_predictability=self.predictability_values,
                                             output_word_frequency_map=frequency_filepath)
        self.frequency_values = frequency_values

        self.lexicon = add_lexicon(words=self.tokens,
                                   lexicon_filepath=lexicon_filepath,
                                   frequency_values=frequency_values,
                                   predictability_values=predictability_values,
                                   include_predicted_without_frequencies=include_predicted_without_frequencies,
                                   save=save_lexicon)

        self.lexicon_word_ngrams = add_lexicon_ngram_mapping(lexicon=self.lexicon,
                                                             bigram_frame=self.bigram_frame,
                                                             ngram_gap=self.ngram_gap)

        self.recognition_thresholds = add_recognition_thresholds(lexicon=self.lexicon,
                                                                 max_threshold=self.max_threshold,
                                                                 max_activity=self.max_activity,
                                                                 freq_weight=self.freq_weight,
                                                                 frequency_values=frequency_values,
                                                                 use_threshold=self.use_threshold)

        self.word_inhibitions = add_word_inhibition_matrix(lexicon=self.lexicon,
                                                           lexicon_word_ngrams=self.lexicon_word_ngrams,
                                                           matrix_filepath=matrix_filepath,
                                                           matrix_parameters_filepath=matrix_parameters_filepath,
                                                           save=save_word_inhibition)

        self.frames = load_frames(frame_filepath)

        time_elapsed = time.perf_counter() - start_time

        logger.info("Time elapsed: " + str(int(time_elapsed)) + " second(s).")
        logger.info('Model ready.')

    def read(self, texts: list[str] = None,
             text_ids: list[str] = None,
             task_name: str = 'reading',
             nr_of_sims: int = 1,
             output_filepath: str = '',
             trials: pd.DataFrame = None,
             verbose: bool = True,
             **kwargs: object) ->list:

        """
        Define the task attributes and initiate model processing of text(s).

        :param texts: list of texts to process.
        Preferably aligned with eye-tracking data which will be used to evaluate simulations.
        :param text_ids: list of text_ids to process. If none, it will be range(0,len(texts)).
        :param task_name: the task name.
        :param nr_of_sims: how many times the model should read the give text(s).
        :param output_filepath: filepath to save the processed text.
        :param trials: information on trials necessary to run the model for flanker task.
        :param verbose: whether to show progress messages in the shell or not.
        :param kwargs: all parameters from TaskAttributes which can be overwritten by the user.
        :return: the simulation output of the model.
        """

        print(f'\nRunning simulations...')

        output = dict()
        start_time = time.perf_counter()

        if verbose: print("Task: " + task_name)

        if not texts:
            texts = self.texts
            if not texts:
                raise Exception('No texts to process. Please provide texts to process either in this function or when creating instance of ReadingModel.')

        if task_name == 'reading':
            task = task_attributes.TaskAttributes(task_name, **kwargs)

            for simulation_id in range(1, nr_of_sims+1):
                simulation_output = dict()
                if text_ids is None: text_ids = range(1, len(texts)+1)

                for text_id, text in zip(text_ids, texts):
                    text_tokens = [pre_process_string(token) for token in text.split(' ') if pre_process_string(token) != '']
                    text_output = read_sequence(self,
                                                task,
                                                text,
                                                text_tokens,
                                                text_id,
                                                verbose=verbose)
                    simulation_output[text_id] = text_output
                    # if filepath is given to save output, save output as the model finishes reading a text
                    if output_filepath:
                        write_out_simulation_data(text_output, output_filepath, simulation_id = simulation_id, text_id = text_id)
                # save output of each simulation
                output[simulation_id] = simulation_output

        elif task_name == 'embedded_words':
            task = task_attributes.EmbeddedWords(task_name, **kwargs)
            self.ngram_to_word_excitation = 1

        elif task_name == 'lex_decis':
            task = task_attributes.Lex_Decis(task_name, **kwargs)
            self.attend_width = 15

            corr = False # default false because error leads to slowing& first trial slow
            trial_data = pd.DataFrame(columns=['sim','stim', 'cond', 'recog wrd','recog RT','LD decis', 'correct','LD RT', 'av. max', 'av. tot','word_evidence','nonw_evidence'])  # data from trial
            for simul_id in range(nr_of_sims):
                for i in range(len(trials)):
                    stim= trials.at[i,task.stimcol]
                    if verbose:
                        print(f'---Trial {i} with stim {stim}---')
                    cycle_data, recog,recog_RT, LD_decis,LD_RT = run_wordrecog(self,
                                                                            task,
                                                                            stim,
                                                                            corr,
                                                                            verbose=verbose)
                    corr = (trials.at[i,task.wordcol] == LD_decis)
                    avMax = cycle_data["max word act"][11]  #[10:30].mean()
                    avTot = cycle_data["tot lex act"][11]   #[10:30].mean()
                    word_ev = cycle_data["word_evidence"][11]
                    nonw_ev = cycle_data["nonw_evidence"][11]

                    if verbose:
                        print(f'    recog. {recog} at {recog_RT-10} with {LD_decis} ({corr}) at {LD_RT-10}.')
                    # now add for the trial recognition & LD outcome + RT, plus counters for simulation and trial
                    trial_data.loc[len(trial_data)] = [simul_id,stim,trials.at[i,task.condcol], recog,recog_RT,LD_decis,corr,LD_RT,avMax,avTot,word_ev,nonw_ev]
            output = trial_data

        elif task_name == 'transposed':
            task = task_attributes.Transposed(task_name, **kwargs)
            self.ngram_to_word_excitation = 2.18
            self.attend_width = 3

        else:
            raise NotImplementedError("Task not implemented. Please choose from 'reading', 'embedded_words', 'flanker', or 'transposed'. Otherwise, please check your task name.")

        time_elapsed = time.perf_counter() - start_time
        time_elapsed = str(round(time_elapsed/60,2)).replace('.','')
        print(f"Duration of simulation runs: {time_elapsed[0]} minute(s) and {time_elapsed[1:]} seconds.")

        return output