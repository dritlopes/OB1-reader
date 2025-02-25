from scipy.spatial import distance
import pandas as pd
import numpy as np
import os
from collections import defaultdict

def generate_embeddings(words, model, tokenizer, device, layer_combi, model_name):

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
                # if layer == 0:
                #     print('Hidden state layer 0: ', layers[layer][0])
                #     print('shape: ', layers[layer][0].shape)
                # if layer == 1:
                #     print('Hidden state layer 1: ', layers[layer][0])
                #     print('shape: ', layers[layer][0].shape)
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

def aggregate_layers(embeddings, layers):

    # aggregate embeddings across layers to get unique representation
    aggregated_embeddings = []
    for sequence_embeddings in embeddings: # loop through each sequence
        all_layers = []
        for n in range(len(layers)): # loop through each layer representation of that sequence
            all_layers.append(sequence_embeddings[n])
        aggregated_embeddings.append(np.mean(np.array(all_layers), axis=0))

    return aggregated_embeddings

def get_similarity(embeddings, word_token_pos_map, sequences, words, model_name, context_type='previous_context'):

    if context_type == 'window':
        # for each text word n, we compute the similarity with n-1, n-2, n-3, n+1, n+2, n+3
        similarities = defaultdict(list)
        # each item in "embeddings" representing a sequence from the text that grows incrementally.
        # e.g. The; The mouse; The mouse was; The mouse was eating; and so on
        # The sequence is actually a number of embeddings, each embedding representing a token in the sequence.
        # print(len(embeddings)) # the number of words in the text
        for i, embedding in enumerate(embeddings):
            # print(i)
            # print(sequences[i])
            # compute fixated word embedding
            token_positions = word_token_pos_map[i]
            # in case word is made of more than one token,
            # the embedding will be the mean of the embeddings of the tokens
            token_embeddings = embedding[token_positions[0]:token_positions[-1] + 1]
            word_embedding = np.mean(token_embeddings, axis=0)
            # compute context embeddings n-3 to n+3
            for n in [1, 2, 3]:
                # previous context (n-1, n-2, n-3)
                if i-1 >= 0:
                    # the context word position should be within the length of the previous context
                    if n in range(1, len(embeddings[i - 1])+1):
                        # take embedding of the previous word
                        context_embedding = embeddings[i - 1][-n]
                        sim = 1 - distance.cosine(word_embedding, context_embedding)
                        similarities[i].append([-n, sim, words[i-n]])
                # upcoming context (n+1, n+2, n+3)
                if i + n < len(embeddings):
                    # take embedding of last token in upcoming sequence
                    context_embedding = embeddings[i+n][-1]
                    sim = 1 - distance.cosine(word_embedding, context_embedding)
                    similarities[i].append([n, sim, words[i+n]])
            # print(similarities[i])
            # print()
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

def calculate_similarity_values(texts_df, corpus_name, model_name, layers, context_type, model, tokenizer, device):

    print(f'Extracting embedding similarity from {model_name}...')

    all_words = []
    for text_id, words in texts_df.groupby('trialid'):
        all_words.append(words['ia'].tolist())

    all_similarity = []
    for i, words in enumerate(all_words):
        print(f"Processing text {i}")
        embeddings, sequences, word_token_pos_map = generate_embeddings(words, model, tokenizer, device, layers, model_name)
        embeddings = aggregate_layers(embeddings, layers)
        assert len(embeddings) == len(sequences), print(len(embeddings), len(sequences))
        similarity = get_similarity(embeddings, word_token_pos_map, sequences, words, model_name, context_type)
        all_similarity.append(similarity)

    if context_type == 'window':
        trial_id, ianum, ia, context_ianum, context_ia, dist, similarity = [], [], [], [], [], [], []
        for id, rows in texts_df.groupby(['trialid', 'ianum', 'ia']):
            text_id = int(id[0])
            word_id = int(id[1])
            word = id[2]
            if text_id < len(all_similarity):
                # print(text_id, word_id, word)
                # it is important that the words and word_ids in the eye-mov data are aligned
                # with the word_ids and words in the similarity data
                # e.g. text id should start at 0
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
        texts_df = pd.DataFrame({'trialid': trial_id,
                                 'ianum': ianum,
                                 'ia': ia,
                                 'context_ianum': context_ianum,
                                 'context_ia': context_ia,
                                 'distance': dist,
                                 'similarity': similarity})

    else:
        all_similarity = [row for text_rows in all_similarity for row in text_rows]
        # print(all_similarity)
        texts_df['similarity'] = all_similarity

    model_name = model_name.replace('/', '_')
    path_to_save = f'../data/{corpus_name}/{model_name}/similarity_{context_type}_{layers}_{model_name}_{corpus_name}_df.csv'
    directory = os.path.dirname(path_to_save)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    texts_df.to_csv(path_to_save, index=False)

    print('DONE!')

    return texts_df