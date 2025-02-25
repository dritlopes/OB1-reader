import numpy as np
import string
import pandas as pd
from torch import nn
import os

# Calculate the surprisal value for each word from original texts (df)
def calculate_surprisal_values(df: pd.DataFrame, corpus_name, model_name, model, tokenizer, device, path_to_save):

    # Process text with language model
    # previous_context = ""  # cumulator, to start a for loop you need an empty variable to include something in each loop
    model_tokens, corpus_tokens = [], [] # lists to save which words in the corpus are multi-tokens in the model
    surprisal_values = []

    print(f'Extracting surprisal values from {model_name}...')
    for text, rows in df.groupby('trialid'):

        previous_context = ''

        for i, next_word in enumerate(rows['ia'].tolist()):

            if i == 0:
                surprisal_values.append(None)  # first word in text does not have context to compute surprisal
                previous_context = next_word

            else:
                next_word = ' ' + next_word
                next_word_clean = next_word.strip(string.punctuation)
                # this line takes the first word and tokenizes it in the form PyTorch
                encoded_input = tokenizer(previous_context, return_tensors='pt').to(device)
                # the list of IDs from the tokenizer
                next_word_id = tokenizer(next_word_clean, return_tensors='pt')["input_ids"][0].to(device)
                model.to(device)
                # get GPT2 output, see https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel for details on the output
                model.eval()  # turn off dropout layers
                output = model(**encoded_input)
                # logits are scores from output layer of shape (batch_size, sequence_length, vocab_size)
                logits = output.logits[:, -1, :]
                # convert raw scores into probabilities (between 0 and 1)
                probabilities = nn.functional.softmax(logits, dim=1)  # softmax transforms the values from logits into percentages
                # take probability of next token in the text (averaging probabilities for multi-token words)
                token_probabs = []
                for token_id in next_word_id:
                    probability = probabilities[0, token_id]
                    probability = probability.cpu().detach().numpy()
                    token_probabs.append(probability)
                probability = np.mean(token_probabs)
                # convert probability into surprisal
                surprisal = -np.log2(probability)
                surprisal_values.append(surprisal)
                # increase context for next surprisal
                previous_context = previous_context + next_word
                # check which words in the corpus are multi-tokens in the model
                if len(next_word_id) > 1:
                    corpus_tokens.append(next_word)
                    model_tokens.append([tokenizer.decode(token_id) for token_id in
                                         next_word_id])

    df['surprisal'] = surprisal_values

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