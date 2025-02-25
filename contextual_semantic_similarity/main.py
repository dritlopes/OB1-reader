import pandas as pd
from process_corpus import pre_process_data
from compute_embeddings import calculate_similarity_values
from compute_surprisal import calculate_surprisal_values
from evaluation import evaluation
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os

language = 'en'
texts_filepath = 'data/raw/Provo/Provo_Corpus-Predictability_Norms.csv'  # 'data/raw/Provo/Provo_Corpus-Predictability_Norms.csv' # 'data/meco/supp_texts.csv'
eye_move_filepath = 'data/raw/Provo/Provo_Corpus-Eyetracking_Data.csv'  # 'data/raw/Provo/Provo_Corpus-Eyetracking_Data.csv' # 'data/meco/joint_data_trimmed.rda'
frequency_filepath = 'data/raw/Provo/SUBTLEX_UK.txt'  # 'data/raw/Provo/SUBTLEX_UK.txt' # 'data/meco/wordlist_meco.csv'
corpus_name = 'meco' # 'Provo' 'meco'
model_name = 'gpt2' # "meta-llama/Llama-2-7b-hf" # gpt2
measures = ['dur', 'skip', 'reread']
compute = True
evaluate = False
n = 5 # number of folds for cross-validation evaluation
seed = 1 # seed to randomly sample trials for cross-validation
context_type = 'previous_context' # 'window' # previous_context
layers = [[i] for i in range(12)] # [[i] for i in range(12)] [[i] for i in range(33)] # [[1]]
model_token = ''
model_name_dir = model_name.replace('/', '_')

if compute:

    # Pre-process stimulus data and eye-movement data
    words_filepath = f'data/processed/{corpus_name}/words_{language}_df.csv'
    processed_eye_move_filepath = f'data/processed/{corpus_name}/corpus_{language}_df.csv'
    if os.path.exists(words_filepath) and os.path.exists(processed_eye_move_filepath):
        words_df = pd.read_csv(words_filepath)
        eye_move_df = pd.read_csv(processed_eye_move_filepath)
    else:
        words_df, eye_move_df = pre_process_data(eye_move_filepath, texts_filepath, frequency_filepath, corpus_name, language, variables=['length','frequency'])
        words_df.to_csv(words_filepath, index=False)
        eye_move_df.to_csv(processed_eye_move_filepath, index=False)

    # Load LM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device ', str(device))
    if 'gpt2' in model_name:
        # see https://huggingface.co/docs/transformers/model_doc/gpt2 for gpt2 documentation
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    elif 'llama' in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, token=model_token)
        model = LlamaForCausalLM.from_pretrained(model_name, token=model_token, torch_dtype=torch.float16).to(device)
    else:
        raise NotImplementedError('Language model not implemented.')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    # Get surprisal values
    surprisal_filepath = f'data/processed/{corpus_name}/{model_name_dir}/surprisal_{model_name_dir}_{corpus_name}_df.csv'
    if os.path.exists(surprisal_filepath):
        surprisal_df = pd.read_csv(surprisal_filepath, sep='\t')
    else:
        surprisal_df = calculate_surprisal_values(words_df, corpus_name, model_name, model, tokenizer, device, surprisal_filepath)

    # Merge eye-mov data and surprisal values
    eye_move_df = pd.merge(eye_move_df, surprisal_df[['trialid', 'ianum', 'surprisal']], how='left', on=['trialid', 'ianum'])

    paths_to_data = [] # store path to files with data from each layer (combination) to be used for evaluation
    # Get similarity values
    for layer_combi in layers:

        similarity_filepath = f'data/processed/{corpus_name}/{model_name_dir}/similarity_{context_type}_{layer_combi}_{model_name}_{corpus_name}_df.csv'
        eye_move_sim_filepath = f'data/processed/{corpus_name}/{model_name_dir}/full_{model_name_dir}_{layer_combi}_{corpus_name}_{context_type}_df.csv'
        paths_to_data.append(eye_move_sim_filepath)

        if os.path.exists(similarity_filepath):
            similarity_df = pd.read_csv(similarity_filepath)
        else:
            print(f"Processing layer(s) {layer_combi}...")
            similarity_df = calculate_similarity_values(words_df,
                                                        model_name,
                                                        layer_combi,
                                                        context_type,
                                                        model,
                                                        tokenizer,
                                                        device,
                                                        similarity_filepath)

        # Merge similarity and eye movement dataframes
        if context_type == 'previous_context':
            eye_move_df = eye_move_df.loc[eye_move_df['ianum'].isin(similarity_df['ianum'].tolist())
                                          & eye_move_df['ia'].isin(similarity_df['ia'].tolist())
                                          & eye_move_df['trialid'].isin(similarity_df['trialid'].tolist())]
            eye_move_sim_df = pd.merge(eye_move_df, similarity_df[['trialid', 'ianum', 'similarity']], how='left', on=['trialid', 'ianum'])
            eye_move_sim_df.to_csv(eye_move_sim_filepath)

if evaluate and context_type == 'previous_context':
    directory = f'data/analysed/{corpus_name}/{model_name}'
    evaluation(layers, paths_to_data, measures, model_name_dir, n, seed, directory)