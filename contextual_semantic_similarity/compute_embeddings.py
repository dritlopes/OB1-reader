from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import pandas as pd
import pickle

def generate_embeddings(words:list[str],
                        model:GPT2LMHeadModel|LlamaForCausalLM,
                        tokenizer:GPT2Tokenizer|LlamaTokenizer,
                        device:torch.device,
                        layer_combi:list[int],
                        model_name:str)-> (torch.Tensor,list,dict):

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
                sequence_vector = layers[layer][0].cpu().detach()
                # sequence_vector = layers[layer][0].cpu().detach().numpy()
                # print(sequence_vector)
                sequence_vectors_per_layer.append(sequence_vector)
            else:
                raise Exception(f'Selected layer {layer} not in language model. Range of hidden layers in language model: {len(layers)}')

        sequence_vectors_per_layer = torch.stack(sequence_vectors_per_layer,dim=0)
        sequence_vectors.append(sequence_vectors_per_layer)

    assert len(sequences) == len(sequence_vectors), print(len(sequences), len(sequence_vectors))

    return sequence_vectors, sequences, word_token_pos_map

def aggregate_layers(embeddings:list[torch.Tensor]) -> list[torch.Tensor]:

    """
    Combine representations from different layers into a single vector for each sequence. If only one layer, return the same vector.
    Args:
        embeddings: list of embeddings

    Returns: averaged vectors

    """

    # aggregate embeddings across layers to get unique representation
    aggregated_embeddings = []
    for sequence_embeddings in embeddings: # loop through each sequence
        # save mean of representation across layers
        aggregated_embeddings.append(torch.mean(sequence_embeddings, dim=0))

    return aggregated_embeddings

def compute_representations(words_df,
                            model,
                            tokenizer,
                            device,
                            layers,
                            model_name,
                            corpus_name):

    # list of list of words (one list per text)
    all_words = []
    for text_id, words in words_df.groupby('trialid'):
        all_words.append(words['ia'].tolist())

    # for each text words
    word_to_token_map = dict()
    for i, words in enumerate(all_words):
        print(f"Processing text {i}")
        # compute word representations
        embeddings, sequences, word_token_pos_map = generate_embeddings(words, model, tokenizer, device, layers, model_name)
        word_to_token_map[i] = word_token_pos_map
        if len(layers) > 0:
            embeddings = aggregate_layers(embeddings)
        assert len(embeddings) == len(sequences), print(len(embeddings), len(sequences))
        # pad sequence vectors to have same size
        padded_vectors = []
        for sequence_embedding in embeddings:
            missing_dimensions = torch.zeros(embeddings[-1].shape[0] - sequence_embedding.shape[0],sequence_embedding.shape[-1])
            padded_embedding = torch.cat((sequence_embedding, missing_dimensions), dim=0)
            padded_vectors.append(padded_embedding)
        all_text_vectors = torch.stack(padded_vectors, dim=0)
        torch.save(all_text_vectors, f'data/processed/{corpus_name}/{model_name}/text{i}_embeddings.pt')
    with open(f'data/processed/{corpus_name}/{model_name}/word_to_token_map.pkl', 'wb') as f:
        pickle.dump(word_to_token_map, f)


def main():

    model_name = 'gpt2'
    model_token = ''
    corpus_name = 'meco'
    layers = [[1]]
    model_name_dir = model_name.replace('/', '_')
    words_filepath = f'data/processed/{corpus_name}/words_en_df.csv'

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

    # load text file
    words_df = pd.read_csv(words_filepath)

    for layer_combi in layers:
        compute_representations(words_df, model, tokenizer, device, layer_combi, model_name, corpus_name)

if __name__ == '__main__':
    main()