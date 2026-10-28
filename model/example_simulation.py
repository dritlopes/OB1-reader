# print('Starting to import')
from model import ReadingModel
from model_components import FixationOutput
from utils import extract_sentences, extract_stimuli
from evaluation import evaluate_reading, evaluate_lex_decis
import task_attributes

text_ids = [1] # range(1,56) # text ids from corpus/stimuli list
trials = []
task_name = 'reading' #'reading', 'flanker'
dataset_name = "provo" # "ziaka25expt1_stim.csv" # meco
language =  'english'  #'english', 'french', 'dutch', 'german'

print('Starting with ' + task_name + ' in ' + language)

if task_name== 'reading':
    # input text_ids for texting list of sentences in the corpus # OR just give a list of strings as texts
    texts = extract_sentences(dataset_name, text_ids = text_ids)
    # OR just give a list of strings as texts
elif task_name== 'lex_decis':
    task = task_attributes.Lex_Decis(task_name) #, **kwargs)
    # input text_ids for selecting trials (and comment out text_ids if all should be run)
    trials = extract_stimuli(dataset_name) #, text_ids = text_ids)
    # texts has to be just words in lex decis because else the nonwords will be entered into the lexicon
    texts = trials.loc[trials[task.wordcol]==task.wordcode, task.stimcol].to_list()
else:
    raise NotImplementedError("Task not yet implemented")

# initialize model with default config

model = ReadingModel(texts=texts,
                     language=language,
                     use_predictability=True,  # switch on predictability-based pre-activation
                     predictability_filepath="../data/processed/prediction_map_Provo_Corpus_gpt2_continuous_reading_english_topkall_0.01.json",
                     frequency_filepath='../data/processed/frequency_map_english.json',
                     # frame_filepath='../data/raw/output_FAST.gemma4-31b.json',  # switch on frame-based activation,:
                     ngram_gap=0,
                     verbose=True)

# run reading simulation
output: list[list[list[FixationOutput]]] = model.read(output_filepath=f'../data/model_output/{model.time}/example_simulation.csv',
                                                      task_name=task_name,
                                                      nr_of_sims=1,
                                                      trials=trials,
                                                      text_ids=text_ids,
                                                      verbose=True)

# evaluate model output
if task_name== 'reading':
    evaluate_reading(output, dataset_name, text_ids=text_ids)
elif task_name== 'lex_decis':
    evaluate_lex_decis(output, f'../data/model_output/flanker_sim{model.time}.csv')