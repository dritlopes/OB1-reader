from scipy.spatial import distance
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.preprocessing import normalize

def generate_embeddings(words:list[str],
                        model:GPT2LMHeadModel|LlamaForCausalLM,
                        tokenizer:GPT2Tokenizer|LlamaTokenizer,
                        device:torch.device,
                        layer_combi:list[int],
                        model_name:str)-> (list,list,dict):

    """
    Extract embeddings for each sequence
    Args:
        words: words from text
        model: gpt2 or llama
        tokenizer: gpt2 or llama tokenizer
        device: cuda or cpu
        layer_combi: list of layer numbers
        model_name: name of language model

    Returns: list of embeddings, list of respective sequences, dictionary with map between (text) word and (language model) token positions

    """

    sequence_vectors, sequences = [], []
    word_token_pos_map = dict()

    for i, word in enumerate(words):
        sequence_vectors_per_layer = []
        sequence = ' '.join(words[:i+1])
        sequences.append(sequence)
        # print(sequence)
        # print(word)
        # use the tokenizer
        # tokens = tokenizer.tokenize(sequence)
        tokens = tokenizer(sequence, return_tensors='pt').to(device)
        # token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # convert the matrix format to torch tensor
        # tokens_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
        tokens_tensor = tokens['input_ids']
        # print(tokens_tensor)

        # find position of last word in token sequence
        if 'gpt' in model_name:
            if i != 0:
                word = ' ' + word
        word_tokens = tokenizer.tokenize(word)
        # print(word_tokens)
        sequence_tokens = tokenizer.tokenize(sequence)
        # print(sequence_tokens)
        # map words to token positions
        word_token_pos_map[i] = [pos for pos in range(len(sequence_tokens)-len(word_tokens), len(sequence_tokens))]

        # Get the model output, see https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput for details
        model.eval()  # turn off dropout layers
        output = model(tokens_tensor, output_hidden_states=True)
        # print(output)
        # Extract the hidden states for all layers
        layers = output.hidden_states
        # hidden_state[0] = word embeddings + positional embedding, hidden_state[-1] = output
        # print(range(len(layers)))

        for layer in layer_combi:
            if layer in range(len(layers)):
                # Our batch consists of a single sequence, so we simply extract the first one
                # The sentence vector is a list of vectors, one for each token in the sequence
                # Each token vector consists of n dimensions
                # print(layers[layer])
                sequence_vector = layers[layer][0].cpu().detach().numpy()
                # print(sequence_vector)
                sequence_vectors_per_layer.append(sequence_vector)
            else:
                raise Exception(f'Selected layer {layer} not in language model. Range of hidden layers in language model: {len(layers)}')
        sequence_vectors.append(sequence_vectors_per_layer)

    assert len(sequences) == len(sequence_vectors), print(len(sequences), len(sequence_vectors))

    return sequence_vectors, sequences, word_token_pos_map

def aggregate_layers(embeddings:list[np.array], layers:list[int]) -> list[np.array]:

    """
    Combine representations from different layers into a single vector for each sequence.
    Args:
        embeddings: list of embeddings
        layers: list of layer numbers

    Returns: averaged vectors

    """

    # aggregate embeddings across layers to get unique representation
    aggregated_embeddings = []
    for sequence_embeddings in embeddings: # loop through each sequence
        all_layers = []
        for n in range(len(layers)): # loop through each layer representation of that sequence
            all_layers.append(sequence_embeddings[n])
        aggregated_embeddings.append(np.mean(np.array(all_layers), axis=0))

    return aggregated_embeddings

def get_similarity(embeddings:list[np.array],
                   word_token_pos_map:dict[int:list],
                   sequences:list[str],
                   words:list[str],
                   model_name:str,
                   context_type='previous_context') -> defaultdict|list:

    """
    Compute similarity between embeddings.
    Args:
        embeddings: the sequence representations pooled from language model
        word_token_pos_map: mapping from word position to token position.
        sequences: sequences with one-word increment
        words: list of words from text
        model_name: name of language model
        context_type: "window" or "previous_context"

    Returns: list of similarity scores (if "previous_context") or a dictionary with word position as key and [context position, similarity score, and context word] as value (if "window")

    """

    if context_type == 'window':
        # for each text word n, we compute the similarity with n-1, n-2, n-3, n+1, n+2, n+3
        similarities = defaultdict(list)
        # each item in "embeddings" representing a sequence from the text that grows incrementally.
        # e.g. The; The mouse; The mouse was; The mouse was eating; and so on
        # The sequence is actually a number of embeddings, each embedding representing a token in the sequence.
        # print(len(embeddings)) # the number of words in the text
        for i, embedding in enumerate(embeddings):
            # print(i, sequences[i])
            # compute fixated word embedding
            token_positions = word_token_pos_map[i]
            # in case word is made of more than one token,
            # the embedding will be the mean of the embeddings of the tokens
            token_embeddings = embedding[token_positions[0]:token_positions[-1] + 1]
            word_embedding = np.mean(token_embeddings, axis=0)
            # compute context embeddings n-3 to n+3
            for n in [-1,-2,-3,1,2,3]:
                # context word position within text
                if 0 <= i + n < len(embeddings):
                    context_word_text_position = i + n
                    # make sure token position is aligned with word position
                    token_positions_context_word = word_token_pos_map[context_word_text_position]
                    if n < 0: # previous context (n-1, n-2, n-3)
                        context_embedding = embeddings[i - 1]
                    else: # upcoming context (n+1, n+2, n+3)
                        context_embedding = embeddings[i + n]
                    context_word_embedding = context_embedding[token_positions_context_word[0]:token_positions_context_word[-1] + 1]
                    # merge embeddings from multi-tokens into one embedding
                    context_word_embedding = np.mean(context_word_embedding, axis=0)
                    sim = 1 - distance.cosine(word_embedding, context_word_embedding)
                    similarities[i].append([n, sim, words[i + n]])
                    # print(n, context_word_text_position, words[context_word_text_position], token_positions_context_word)

    else:
        # for each text word, we compute the similarity with its previous context
        similarities = []
        for i, embedding in enumerate(embeddings):
            if i-1 in range(len(embeddings)):
                # get embedding of word based on the equivalent token(s) position(s)
                # print(i)
                # print(sequences[i])
                token_positions = word_token_pos_map[i]
                # print(token_positions)
                # print(embedding.shape)
                if 'gpt' in model_name:
                    token_embeddings = embedding[token_positions[0]:token_positions[-1] + 1]
                else: # llama (bcs of BOS token)
                    token_embeddings = embedding[token_positions[0]+1:token_positions[-1] + 2]
                # print(token_positions[0], token_positions[-1]+1)
                # print(token_embeddings)
                # aggregate token embeddings that form the word (if more than one token, take the average)
                word_embedding = np.mean(token_embeddings, axis=0)
                # print(word_embedding)
                # if the word is equivalent to one token, token embedding and word embedding should be the same
                if len(token_embeddings) == 1: assert list(token_embeddings[0]) == list(word_embedding)
                # context embedding
                # aggregate embeddings of tokens in previous context
                context_embedding = np.mean(embeddings[i-1], axis=0)
                # print(i)
                # print('Context embedding', context_embedding)
                # print('shape: ', context_embedding.shape)
                # print('Word embedding', word_embedding)
                # print('shape: ', word_embedding.shape)
                # print('Division result: ', np.divide(context_embedding, word_embedding))
                # print(embeddings[i-1])
                # print(embeddings[i-1].shape)
                # print()
                # compute sim between word and previous context representations
                sim = 1 - distance.cosine(word_embedding, context_embedding)
                # print('Similarity', sim)
                # print(sim)
            else:
                sim = None
            similarities.append(sim)
            # print(sim)
        assert len(similarities) == len(sequences), print(len(similarities), len(sequences))

    return similarities

def calculate_similarity_values(words_df: pd.DataFrame,
                                model_name:str,
                                layers:list[int],
                                context_type:str,
                                model:GPT2LMHeadModel|LlamaForCausalLM,
                                tokenizer:GPT2Tokenizer|LlamaTokenizer,
                                device:torch.device,
                                path_to_save:str) -> pd.DataFrame:

    """
    Extract embeddings and calculate similarity scores based on embeddings.
    Args:
        words_df: words dataframe
        model_name: name of language model
        layers: list of hidden layers from which to extract embeddings to compute similarity
        context_type: "window" or "previous_context"
        model: gpt2 or llama
        tokenizer: gpt2 or llama tokenizer
        device: cuda or cpu
        path_to_save: path to similarity scores

    Returns: dataframe with similarity scores

    """

    print(f'Extracting embedding similarity from {model_name}...')

    # list of list of words (one list per text)
    all_words = []
    for text_id, words in words_df.groupby('trialid'):
        all_words.append(words['ia'].tolist())

    # extract embeddings and compute similarity
    all_similarity = []
    for i, words in enumerate(all_words):
        print(f"Processing text {i}")
        # compute word representations
        embeddings, sequences, word_token_pos_map = generate_embeddings(words, model, tokenizer, device, layers, model_name)
        if len(layers) > 0:
            embeddings = aggregate_layers(embeddings, layers)
        assert len(embeddings) == len(sequences), print(len(embeddings), len(sequences))
        # compute similarity scores
        similarity = get_similarity(embeddings, word_token_pos_map, sequences, words, model_name, context_type)
        all_similarity.append(similarity)

    # if context_type is window
    if context_type == 'window':
        trial_id, ianum, ia, context_ianum, context_ia, dist, similarity = [], [], [], [], [], [], []
        for id, rows in words_df.groupby(['trialid', 'ianum', 'ia']):
            text_id = int(id[0])
            word_id = int(id[1])
            word = id[2]
            if text_id < len(all_similarity):
                # print(text_id, word_id, word)
                # it is important that the words and word_ids in the eye-mov data are aligned
                # with the word_ids and words in the similarity data
                similarities = all_similarity[text_id][word_id]
                # similarity score between word and each context word (n-1,n-2,n-3,n+1,n+2,n+3)
                for sim in similarities:
                    trial_id.append(text_id)
                    ianum.append(word_id)
                    ia.append(word)
                    context_ianum.append(word_id+sim[0])
                    context_ia.append(sim[2])
                    dist.append(sim[0])
                    similarity.append(sim[1])
        words_df = pd.DataFrame({'trialid': trial_id,
                                 'ianum': ianum,
                                 'ia': ia,
                                 'context_ianum': context_ianum,
                                 'context_ia': context_ia,
                                 'distance': dist,
                                 'similarity': similarity})

    # if context type is previous context, simply add similarity scores as a column to the words data
    else:
        all_similarity = [row for text_rows in all_similarity for row in text_rows]
        # print(all_similarity)
        words_df['similarity'] = all_similarity

    # save dataframe with similarity score
    directory = os.path.dirname(path_to_save)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    words_df.to_csv(path_to_save, index=False)

    print('DONE!')

    return words_df

def merge_eye_sim_window(similarity_df:pd.DataFrame, eye_move_df:pd.DataFrame) -> pd.DataFrame:

    """
    Merge eye movement data and window similarity data.
    Args:
        similarity_df: dataframe containing similarity scores.
        eye_move_df: dataframe containing eye movement data.

    Returns: dataframe containing eye movement data and similarity scores.
    """

    # add fix id
    fix_ids = []
    for i, rows in eye_move_df.groupby(['participant_id','trialid']):
        fix_ids.extend([i for i in range(len(rows))])

    eye_move_df['fixid'] = fix_ids
    # filter fixations with saccades within window size -3 to +3
    eye_move_df_filtered = eye_move_df.loc[eye_move_df['next_saccade_distance'].isin([-3, -2, -1, 0, 1, 2, 3])].copy()
    window_dict = defaultdict(list)

    # iter through each fixation
    for i in eye_move_df_filtered.itertuples():
        previous_sacc_distance, previous_fix_duration = None, None
        # find saccade distance
        saccade_distance = i.next_saccade_distance
        previous_fix = eye_move_df[(eye_move_df['participant_id']==i.participant_id) & (eye_move_df['trialid']==i.trialid) & (eye_move_df['fixid']==i.fixid-1)]
        if not previous_fix.empty:
            # find previous saccade distance
            previous_sacc_distance = previous_fix['next_saccade_distance'].tolist()[0]
            # find previous fixation duration
            previous_fix_duration = previous_fix['dur'].tolist()[0]
        # iter through each context word of fixated word
        context = similarity_df[(similarity_df['trialid'] == i.trialid) & (similarity_df['ianum'] == i.ianum)]
        # filter contexts with all six positions
        if len(context) == 6:

            # Add fixated word to context words to account for refixations
            window_dict['participant_id'].append(i.participant_id)
            window_dict['trialid'].append(i.trialid)
            window_dict['fixid'].append(i.fixid)
            window_dict['ianum'].append(i.ianum)
            window_dict['ia'].append(i.ia)
            window_dict['letternum'].append(i.letternum)
            window_dict['letter'].append(i.letter)
            window_dict['previous_sacc_distance'].append(previous_sacc_distance)
            window_dict['previous_fix_duration'].append(previous_fix_duration)
            window_dict['context_ianum'].append(i.ianum)
            window_dict['context_ia'].append(i.ia)
            window_dict['length'].append(len(i.ia))
            window_dict['frequency'].append(i.frequency)
            window_dict['pos_tag'].append(i.pos_tag)
            window_dict['surprisal'].append(i.surprisal)
            window_dict['entropy'].append(i.entropy)
            window_dict['similarity'].append(1)
            window_dict['distance'].append(0)
            if saccade_distance == 0:
                # only for the actual landing target, the rest none
                window_dict['letter_distance'].append(i.next_saccade_letter_distance)
                window_dict['landing_target'].append(1)
            else:
                window_dict['letter_distance'].append(None)
                window_dict['landing_target'].append(0)

            # Add each context word
            for context_word in context.itertuples():
                # add other word context variables
                saccade_target, letter_distance = 0, None
                frequency, pos_tag, surprisal, entropy = None, None, None, None
                # saccade target and letter distance
                if saccade_distance == context_word.distance:
                    saccade_target = 1
                    letter_distance = i.next_saccade_letter_distance
                # frequency, postag, surprisal and entropy
                rows = eye_move_df_filtered.loc[(eye_move_df_filtered['trialid'] == context_word.trialid) & (
                        eye_move_df_filtered['ianum'] == context_word.context_ianum)]
                if 'frequency' in eye_move_df_filtered.columns:
                    if len(rows['frequency'].tolist()) > 0:
                        frequency = rows['frequency'].tolist()[0]
                if 'pos_tag' in eye_move_df_filtered.columns:
                    if len(rows['pos_tag'].tolist()) > 0:
                        pos_tag = rows['pos_tag'].tolist()[0]
                if 'surprisal' in eye_move_df_filtered.columns:
                    if len(rows['surprisal'].tolist()) > 0:
                        surprisal = rows['surprisal'].tolist()[0]
                if 'entropy' in eye_move_df_filtered.columns:
                    if len(rows['entropy'].tolist()) > 0:
                        entropy = rows['entropy'].tolist()[0]
                window_dict['participant_id'].append(i.participant_id)
                window_dict['trialid'].append(i.trialid)
                window_dict['fixid'].append(i.fixid)
                window_dict['ianum'].append(i.ianum)
                window_dict['ia'].append(i.ia)
                window_dict['letternum'].append(i.letternum)
                window_dict['letter'].append(i.letter)
                window_dict['previous_sacc_distance'].append(previous_sacc_distance)
                window_dict['previous_fix_duration'].append(previous_fix_duration)
                window_dict['context_ianum'].append(context_word.context_ianum)
                window_dict['context_ia'].append(context_word.context_ia)
                window_dict['length'].append(len(context_word.context_ia))
                window_dict['frequency'].append(frequency)
                window_dict['pos_tag'].append(pos_tag)
                window_dict['surprisal'].append(surprisal)
                window_dict['entropy'].append(entropy)
                window_dict['similarity'].append(context_word.similarity)
                window_dict['distance'].append(context_word.distance)
                window_dict['letter_distance'].append(letter_distance)
                window_dict['landing_target'].append(saccade_target)

    df = pd.DataFrame.from_dict(window_dict)
    df.sort_values(by=['participant_id','trialid','fixid','distance'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def main():

    """
    Given a language model and an eye-tracking corpus, compute semantic similarity for (fixated) word in the corpus.
    Returns: saves out the similarity values, and eye-tracking data with similarity values added.
    """

    model_name = 'gpt2' # meta-llama_Llama-2-7b-hf
    corpus_name = 'meco' # provo
    context_type = 'window' # previous_context
    layers = [[1]]
    model_token = '' # needed if llama is the language model
    model_name_dir = model_name.replace('/', '_')
    words_filepath = f'data/processed/{corpus_name}/words_en_df.csv' # path to file with word dataset
    eye_move_filepath = f'data/processed/{corpus_name}/{model_name_dir}/fixation_report_{corpus_name}_surprisal_{model_name_dir}.csv' # path to eye-tracking dataset

    # load word and eye-movement dataframes
    words_df = pd.read_csv(words_filepath)
    eye_move_df = pd.read_csv(eye_move_filepath)

    # for each langauge model layer or combination of layers
    for layer_combi in layers:

        similarity_filepath = f'data/processed/{corpus_name}/{model_name_dir}/similarity_{context_type}_{layer_combi}_{model_name}_{corpus_name}_df.csv'
        eye_move_sim_filepath = f'data/processed/{corpus_name}/{model_name_dir}/full_{model_name_dir}_{layer_combi}_{corpus_name}_{context_type}_df.csv'

        # if dataset with similarity scores already exists, simply read it in
        if os.path.exists(similarity_filepath):
            similarity_df = pd.read_csv(similarity_filepath)

        # else compute similarity values for each word and its context (either window condition or previous context condition)
        else:
            print('Extracting semantic similarity values for text data...')
            # load LM
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('Using device ', str(device))
            if 'gpt2' in model_name:
                # see https://huggingface.co/docs/transformers/model_doc/gpt2 for gpt2 documentation
                model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            elif 'llama' in model_name:
                tokenizer = LlamaTokenizer.from_pretrained(model_name, token=model_token)
                model = LlamaForCausalLM.from_pretrained(model_name, token=model_token, torch_dtype=torch.float16).to(
                    device)
            else:
                raise NotImplementedError('Language model not implemented.')
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

            # compute semantic similarity scores
            print(f"Processing layer(s) {layer_combi}...")
            similarity_df = calculate_similarity_values(words_df,
                                                        model_name,
                                                        layer_combi,
                                                        context_type,
                                                        model,
                                                        tokenizer,
                                                        device,
                                                        similarity_filepath)

        # merge similarity and eye movement dataframes
        print('Combining semantic similarity values with eye-movement data...')
        if context_type == 'previous_context':
            eye_move_df_filtered = eye_move_df.loc[eye_move_df['ianum'].isin(similarity_df['ianum'].tolist())
                                          & eye_move_df['ia'].isin(similarity_df['ia'].tolist())
                                          & eye_move_df['trialid'].isin(similarity_df['trialid'].tolist())]
            eye_move_sim_df = pd.merge(eye_move_df_filtered, similarity_df[['trialid', 'ianum', 'similarity']], how='left',
                                       on=['trialid', 'ianum'])
            eye_move_sim_df.to_csv(eye_move_sim_filepath)
        elif context_type == 'window':
            eye_move_sim_df = merge_eye_sim_window(similarity_df, eye_move_df)
            eye_move_sim_df.to_csv(eye_move_sim_filepath)

if __name__ == '__main__':
    main()