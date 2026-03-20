# The OB1-reader 

### A model of eye movements during reading

This repository contains the code to run the OB1-reader (first proposed by Snell et al. 2018), a model that simulates word recognition and eye movement control during reading. The model is a computational implementation of cognitive theories on reading and can be used to test predictions of these theories on various experimental set-ups. Previous studies have shown both quantitative and qualitative fits to human behavioural data in various tasks (e.g. Snell et al. 2018 for natural reading, Meeter et al. 2020 for flankers and priming).

The theoretical framework OB1-reader formulates primarily results from integrating relative letter-position coding (Grainger & Van Heuven, 2004) and parallel graded processing (e.g. Engbert et al. 2005) to bridge accounts on single-word recognition and eye movements during reading, respectively. These are largely based on bottom-up, automatic and early processing of visual input. Because language processing during reading is not limited to bottom-up processing and word recognition, we are currently working on expanding the model to contain feedback processes at the semantic and discourse levels to simulate reading comprehension in various dimensions: as a process, as a product and as a skill. This is the main goal of the PhD project entitled *Computational Models of Reading Comprehension.*

Check out some of the work done as part of this PhD project:

* Lopes Rego, A. T., Snell, J., & Meeter, M. (2024). Language models outperform cloze predictability in a cognitive model of reading. *PLOS Computational Biology, 20(9)*. To reproduce experiments from paper: https://github.com/dritlopes/language_models_outperform_cloze_predictability_in_a_cognitive_model_of_reading

* Lopes Rego, A. T., Snell, J., & Meeter, M. (2025). What determines where readers fixate next? Leveraging NLP to investigate human cognition. In *Proceedings of the First International Workshop on Gaze Data and Natural Language Processing* (pp. 1-6). To reproduce experiments from paper: https://github.com/dritlopes/OB1-reader/tree/contextual_semantic_similarity

* Lopes Rego, A. T., Nogueira, A., & Meeter, M. (2026). What Drives Regressions in Reading? Insights from Surprisal and Saliency from Language Models. \[Manuscript submitted for publication]. To reproduce experiments from paper: https://github.com/dritlopes/modeling_regressions_with_surprisal_and_saliency

## How to run the code

1. To run simulations with OB1-reader, you need input text(s), which you can either provide as a file or as a list of strings directly as an argument to the model (see in example code below). 
2. You wil also need resources for frequency and predictability of the words in the input text(s). Place the resources in the directory `data/raw` (see Repository Structure below for more information).

   * Frequencies for English can be `SUBTLEX_UK`, which can be found at https://osf.io/zq49t/.
   * If your input texts are from MECO, use `joint_fix_trimmed<.rda/.csv>,` which contains the fixation report to evaluate the simulations, and `wordlist_meco.csv` for the word frequency values. All files can be obtained at https://osf.io/3527a/.
   * If your input texts are from Provo, and you'd like to use Provo's cloze predictability values, use `Provo_Corpus-Predictability_Norms.csv`, which can be found at https://osf.io/sjefs/.

3. Example code for running simulations with OB1-reader:

```
from model import ReadingModel
from model_components import FixationOutput
from evaluation import evaluate, extract_sentences

# if texts are from the corpus "meco" or the corpus "provo"
dataset_name = "provo"
# text ids from eye-tracking corpus
text_ids = [1,2,3]
# input texts from corpus
texts = extract_sentences(dataset_name, text_ids = text_ids)

# OR optionally give any list of strings as texts
# texts = ['This is a cool model', 'OB1-reader loves to read']

# initialize model with default config
model = ReadingModel(texts, verbose=True)

# run reading simulation
output:list[list[list[FixationOutput]]] = model.read(output_filepath=f'../data/model_output/{model.time}/example_simulation.csv', 
                                                     number_of_simulations=2, 
                                                     verbose=False)

# evaluate model output against eye-tracking data with the same texts
evaluate(output, dataset_name, text_ids = text_ids)
```

4. In order to evaluate the model, please make sure you have the eye-tracking data on the same texts. We currently support evaluation for the Provo and MECO corpus. For Provo, make sure you have `../data/raw/Provo_Corpus-Eyetracking_Data.csv`; For MECO, make sure you have `../data/raw/joint_data_trimmed.csv`

## Repository structure

### `/data`

This folder should contain all data needed for running simulations. It is made of four sub-folders:

* `/raw`: place text data (e.g. Provo (Luke & Christianson, 2018) and resources for word frequency and word predictability (e.g. SUBTLEX-UK (Van Heuven et al., 2014)) from other sources.
* `/processed`: all the pre-processed data to be used in the model simulations (e.g. eye-movement data cleaned and re-aligned; predictability values from the language models stored as a look-up dictionary).
* `/model_output`: all output of the model simulations (fixation-centered data).
* `/eval_output`: all data resulting from analysing or evaluating the output of the model simulations.

### `/model`

This folder contains the scrips used to run the simulations.

* `model.py`: where the model class is defined. 
* `model_components.py`: the major functions corresponding to sub-processes in word recognition and eye movement are defined. These sub-processes include word activity computation, slot matching and saccade programming.
* `reading_helper_functions.py`: contains helper functions for the processing during reading simulation, e.g. calculating a letter's visual accuity.
* `task_attributes.py`: where the attributes of the specific reading task are defined.
* `fixation_to_words.py`: define classes and functions to store output data in structure way.
* `evaluation.py`: functions to evaluate model output against eye-tracking data.
* `utils.py`: contains helper functions for setting up the model and reading simulations, e.g. set up word frequency data to be used by model; write out simulation data.
* `example_stimulation.py`: example on how to run reading simulation with OB1-reader.

## References

* Snell, J., van Leipsig, S., Grainger, J., & Meeter, M. (2018). OB1-reader: A model of word recognition and eye movements in text reading. Psychological review, 125(6), 969.
* Meeter, M., Marzouki, Y., Avramiea, A. E., Snell, J., & Grainger, J. (2020). The role of attention in word recognition: Results from OB1‐reader. Cognitive science, 44(7), e12846.
* Grainger, J., & Van Heuven, W. J. (2004). Modeling letter position coding in printed word perception.
* Engbert, R., Nuthmann, A., Richter, E. M., & Kliegl, R. (2005). SWIFT: a dynamical model of saccade generation during reading. Psychological review, 112(4), 777.
* Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with predictability norms. Behavior research methods, 50, 826-833.
* Van Heuven, W. J., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK: A new and improved word frequency database for British English. Quarterly journal of experimental psychology, 67(6), 1176-1190.