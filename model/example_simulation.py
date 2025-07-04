from model import ReadingModel
from model_components import FixationOutput
from evaluation import evaluate, extract_sentences

# "meco" or "provo"
dataset_name = "meco"

# input text_ids for texting list of sentences in the corpus
texts = extract_sentences(dataset_name)
# texts = ['In ancient Roman religion and myth, Janus is the god of beginnings and gates.']

# initialize model with default config
model = ReadingModel(texts,
                     predictability_filepath='../data/processed/prediction_map_meco_gpt2_english_topkall_0.01.json',
                     frequency_filepath='../data/processed/frequency_map_english.json',
                     save_lexicon=True,
                     save_word_inhibition=True,
                     lexicon_filepath='../data/processed/lexicon.json',
                     matrix_filepath='../data/processed/inhibition_matrix_previous.pkl',
                     matrix_parameters_filepath='../data/processed/inhibition_matrix_parameters_previous.pkl')

# run reading simulation
output:list[list[list[FixationOutput]]] = model.read(output_filepath=f'../data/model_output/{model.time}/{dataset_name}.csv',
                                                     number_of_simulations=10,
                                                     verbose=False)

# evaluate model output
# evaluate(output, dataset_name)
