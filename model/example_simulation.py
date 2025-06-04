from model import ReadingModel
from model_components import FixationOutput
from evaluation import evaluate, extract_sentences

# text ids from eye-tracking corpus
text_ids = [1,2,3]
# "meco" or "provo"
dataset_name = "provo"

# input text_ids for texting list of sentences in the corpus
texts = extract_sentences(dataset_name, text_ids = text_ids)

# initialize model with default config
model = ReadingModel(texts, ngram_gap=0)

# run reading simulation
output:list[list[list[FixationOutput]]] = model.read(output_filepath=f'../data/model_output/{model.time}/example_simulation.csv', number_of_simulations=2, verbose=False)

# evaluate model output
evaluate(output, dataset_name, text_ids = text_ids)
