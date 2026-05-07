print('Starting to import')
from model import ReadingModel
from model_components import FixationOutput
from evaluation import evaluate, extract_sentences, extract_stimuli

# text ids from eye-tracking corpus
text_ids = [1,2]

task_name = 'reading' #'reading', 'flanker'
language =  'english'  #'english', 'dutch', 'german'

print('Starting with ' + task_name + ' in ' + language)

if task_name== 'reading':
    # "meco" or "provo"
    dataset_name = "provo"
    # input text_ids for texting list of sentences in the corpus
    texts = extract_sentences(dataset_name, text_ids = text_ids)
    # OR just give a list of strings as texts
elif task_name== 'flanker':
    # give file name
    dataset_name = "stim_Flanker_French.csv"
    # input text_ids for texting list of sentences in the corpus
    trials = extract_stimuli(dataset_name, text_ids = text_ids)
    texts = trials["stimulus"].to_list()
    print(texts)
else:
    raise NotImplementedError("Task not yet implemented")

# initialize model with default config
model = ReadingModel(texts, language=language, verbose=True)

print('Model ready...')

# run reading simulation
output:list[list[list[FixationOutput]]] = model.read(output_filepath=f'../data/model_output/{model.time}/example_simulation.csv', task_name = task_name, nr_of_sims=2, verbose=True)

# evaluate model output
evaluate(output, dataset_name, text_ids = text_ids)