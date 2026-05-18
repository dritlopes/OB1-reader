print('Starting to import')
from model import ReadingModel
from model_components import FixationOutput
from evaluation import evaluate, evaluate_task, extract_sentences, extract_stimuli
import task_attributes

# text ids from eye-tracking corpus
text_ids = range(1, 3) # [1,2]
trials =[]

task_name = 'reading' #'reading', 'flanker'
language =  'english'  #'english', 'french', 'dutch', 'german'

print('Starting with ' + task_name + ' in ' + language)

if task_name== 'reading':
    dataset_name = "provo" # "meco" or "provo"
    # input text_ids for texting list of sentences in the corpus
    texts = extract_sentences(dataset_name, text_ids = text_ids)
    # OR just give a list of strings as texts
elif task_name== 'flanker':
    dataset_name = "stim_Flanker_French.csv" # name of file in data/raw
    task = task_attributes.Flanker(task_name) #, **kwargs)
    # input text_ids for selecting trials
    trials = extract_stimuli(dataset_name, text_ids = text_ids)
    # texts has to be just words in lex decis because else the nonwords will be entered into the lexicon
    texts = trials.loc[trials[task.wordcol]==task.wordcode, task.stimcol].to_list()
    print(texts)
else:
    raise NotImplementedError("Task not yet implemented")

# initialize model with default config
model = ReadingModel(texts, language=language, verbose=True)

print('Model ready...')

# run reading simulation
output: list[list[list[FixationOutput]]] = model.read(
    output_filepath=f'../data/model_output/{model.time}/example_simulation.csv', task_name=task_name, nr_of_sims=1, trials=trials,
    verbose=True)

# evaluate model output
if task_name== 'reading':
    evaluate(output, dataset_name, text_ids=text_ids)
else:
    evaluate_task(output,task_name, f'../data/model_output/flanker_sim{model.time}.csv')