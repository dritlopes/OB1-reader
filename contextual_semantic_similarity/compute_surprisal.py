import numpy as np
import pandas as pd
from torch import nn
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from scipy.stats import entropy

def calculate_surprisal_values(df: pd.DataFrame,
                               model_name:str,
                               model:GPT2LMHeadModel|LlamaForCausalLM,
                               tokenizer:GPT2Tokenizer|LlamaTokenizer,
                               device:torch.device,
                               path_to_save:str)->pd.DataFrame:

    """
    # Calculate the surprisal value for each word from corpus texts.
    Args:
        df: dataframe with words from corpus. It contains the text ids, the word ids, and the words.
        model_name: name of language model.
        model: gpt2 or llama model.
        tokenizer: gpt2 or llama tokenizer.
        device: cuda or cpu.
        path_to_save: path to save the surprisal values.

    Returns: dataframe with surprisal values.

    """

    # lists to save which words in the corpus are multi-tokens in the model
    model_tokens, corpus_tokens = [], []
    # list to save surprisal values
    surprisal_values = []
    # list to save entropy values
    entropy_values = []

    print(f'Extracting surprisal values from {model_name}...')
    for text, rows in df.groupby('trialid'):

        previous_context = ''

        for i, next_word in enumerate(rows['ia'].tolist()):

            if i == 0:
                # first word in text does not have context to compute surprisal
                surprisal_values.append(None)
                entropy_values.append(None)
                previous_context = next_word

            else:
                next_word = ' ' + next_word
                # next_word_clean = next_word.strip(string.punctuation)
                # tokenize previous context
                encoded_input = tokenizer(previous_context, return_tensors='pt').to(device)
                # tokenize word
                next_word_id = tokenizer(next_word, return_tensors='pt')["input_ids"][0].to(device)
                model.to(device)
                # turn off dropout layers
                model.eval()
                output = model(**encoded_input)
                # logits are scores from output layer of shape (batch_size, sequence_length, vocab_size)
                logits = output.logits[:, -1, :]
                # convert raw scores into probabilities (between 0 and 1)
                probabilities = nn.functional.softmax(logits, dim=1)  # softmax transforms the values from logits into percentages
                # compute entropy of probability distributions
                entropy_score = entropy(probabilities[0].cpu().detach().numpy())
                entropy_values.append(entropy_score)
                # take surprisal of word in the text (summing surprisal for multi-token words)
                token_surprisal = []
                for token_id in next_word_id:
                    probability = probabilities[0, token_id]
                    probability = probability.cpu().detach().numpy()
                    # convert probability into surprisal
                    surprisal = -np.log2(probability)
                    token_surprisal.append(surprisal)
                surprisal = np.sum(token_surprisal)
                surprisal_values.append(surprisal)
                # increase context for next surprisal
                previous_context = previous_context + next_word
                # check which words in the corpus are multi-tokens in the model
                if len(next_word_id) > 1:
                    corpus_tokens.append(next_word)
                    model_tokens.append([tokenizer.decode(token_id) for token_id in
                                         next_word_id])

    df['surprisal'] = surprisal_values
    df['entropy'] = entropy_values

    # write out surprisal dataset
    directory = os.path.dirname(path_to_save)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    df.to_csv(path_to_save, sep='\t')

    # write out which words in the corpus are multi-tokens in the model
    path_to_save_multi_tokens = path_to_save.replace('.csv', '_multi_tokens.csv')
    with open(path_to_save_multi_tokens, 'w') as outfile:
        outfile.write(f'CORPUS_TOKEN\tMODEL_TOKEN\n')
        for model_token, corpus_token in zip(model_tokens, corpus_tokens):
            outfile.write(f'{corpus_token}\t{model_token}\n')

    return df

def main():

    """
    Given a language model and an eye-tracking corpus, compute surprisal values for each word in the corpus.
    Returns: saves out the surprisal values for each word in the corpus, and the eye-tracking corpus file with surprisal values (as an extra column to the eye-movement data).
    """

    corpus_name = 'meco' # provo
    model_name = 'gpt2' # meta-llama_Llama-2-7b-hf
    model_name_dir = model_name.replace('/', '_') # needed if model is llama
    model_token = '' # needed if model is llama
    words_filepath = f'data/processed/{corpus_name}/words_en_df.csv'  # word dataset which we compute surprisal values for
    eye_move_filepath = f'data/processed/{corpus_name}/fixation_report_en_df.csv'  # corpus dataset with eye-movement data
    surprisal_filepath = f'data/processed/{corpus_name}/{model_name_dir}/surprisal_{model_name_dir}_{corpus_name}_df.csv' # where to save the surprisal values
    eye_surprisal_filepath = f'data/processed/{corpus_name}/{model_name_dir}/fixation_report_{corpus_name}_surprisal_{model_name_dir}.csv' # where to save corpus data set with surprisal values

    # if surprisal dataframe already exists, simply read it in
    if os.path.exists(surprisal_filepath):
        surprisal_df = pd.read_csv(surprisal_filepath)

    # otherwise, compute surprisal values
    else:
    # load LM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device ', str(device))
        if 'gpt2' in model_name:
            model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        elif 'llama' in model_name:
            tokenizer = LlamaTokenizer.from_pretrained(model_name, token=model_token)
            model = LlamaForCausalLM.from_pretrained(model_name, token=model_token, torch_dtype=torch.float16).to(
                device)
        else:
            raise NotImplementedError('Language model not implemented.')
        # if model loaded in gpu, print out memory usage
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        # read in word dataset
        words_df = pd.read_csv(words_filepath)
        # compute surprisal values using language model and word dataset
        surprisal_df = calculate_surprisal_values(words_df, model_name, model, tokenizer, device, surprisal_filepath)

    # Merge eye-mov data and surprisal values
    eye_move_df = pd.read_csv(eye_move_filepath)
    eye_move_df = pd.merge(eye_move_df, surprisal_df[['trialid', 'ianum', 'surprisal', 'entropy']], how='left', on=['trialid', 'ianum'])
    eye_move_df.to_csv(eye_surprisal_filepath, index=False)

if __name__ == '__main__':
    main()