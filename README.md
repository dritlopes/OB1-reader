# The OB1-reader 
### A model of eye movements during reading

This repository contains the code to run the OB1-reader (first proposed by Snell et al. 2018), a model that simulates word recognition and eye movement control during reading. The model is a computational implementation of cognitive theories on reading and can be used to test predictions of these theories on various experimental set-ups. Previous studies have shown both quantitative and qualitative fits to human behavioural data in various tasks (e.g. Snell et al. 2018 for natural reading, Meeter et al. 2020 for flankers and priming).

The theoretical framework OB1-reader formulates primarily results from integrating relative letter-position coding (Grainger & Van Heuven, 2004) and parallel graded processing (e.g. Engbert et al. 2005) to bridge accounts on single-word recognition and eye movements during reading, respectively. These are largely based on bottom-up, automatic and early processing of visual input. Because language processing during reading is not limited to bottom-up processing and word recognition, we are currently working on expanding the model to contain feedback processes at the semantic and discourse levels to simulate reading comprehension in various dimensions: as a process, as a product and as a skill. This is the main goal of the PhD project entitled *Computational Models of Reading Comprehension.*

Check out some of the work done as part of this PhD project:

* *Lopes Rego, A. T., Snell, J., & Meeter, M. (2024). Language models outperform cloze predictability in a cognitive model of reading. PLOS Computational Biology, 20(9).* To reproduce experiments from paper: https://github.com/dritlopes/language_models_outperform_cloze_predictability_in_a_cognitive_model_of_reading

* *Lopes Rego, A. T., Nogueira, A., & Meeter, M. (2025). Language models capture where readers look back to in a text. (in prep.)* To reproduce experiments from paper: https://github.com/dritlopes/modeling_regressions_with_surprisal_and_saliency

## How to run the code

```
from model import ReadingModel

text = ["There are now rumblings that Apple might soon invade the smart watch space, though the company is maintaining its customary silence. "
        "The watch doesn't have a microphone or speaker, but you can use it to control the music on your phone. "
        "You can glance at the watch face to view the artist and title of a song."]

# initialize model with default config
model = ReadingModel(text)

# run reading simulation and write out output to specific filepath
output = model.read(output_filepath=f'../data/model_output/{model.time}/example_simulation.csv')
```

* TODO add functions for post-processing model output (more meaningful eye-movement measures)
* TODO add evaluation scripts for supported corpora (Provo and MECO so far)
* TODO add pre-processing scripts for supported corpora (Provo and MECO so far)

## Repository structure

### `/data`

This folder should contain all data needed for running simulations. It is made of four sub-folders:

* `/raw`: place text data (e.g. Provo (Luke & Christianson, 2018) and resources for word frequency and word predictability (e.g. SUBTLEX-UK (Van Heuven et al., 2014)) from other sources.
* `/processed`: all the pre-processed data to be used in the model simulations (e.g. eye-movement data cleaned and re-aligned; predictability values from the language models stored as a look-up dictionary).
* `/model_output`: all output of the model simulations (fixation-centered data).
* `/analysed`: all data resulting from analysing or evaluating the output of the model simulations.

### `/model`

This folder contains the scrips used to run the simulations.

* `model.py`: where the model class is defined. 
* `model_components.py`: the major functions corresponding to sub-processes in word recognition and eye movement are defined. These sub-processes include word activity computation, slot matching and saccade programming.
* `reading_helper_functions.py`: contains helper functions for the processing during reading simulation, e.g. calculating a letter's visual accuity.
* `task_attributes.py`: where the attributes of the specific reading task are defined.
* `utils.py`: contains helper functions for setting up the model and reading simulations, e.g. set up word frequency data to be used by model; write out simulation data.
* `example_stimulation.py`: example on how to run reading simulation with OB1-reader.

## References

* Snell, J., van Leipsig, S., Grainger, J., & Meeter, M. (2018). OB1-reader: A model of word recognition and eye movements in text reading. Psychological review, 125(6), 969.
* Meeter, M., Marzouki, Y., Avramiea, A. E., Snell, J., & Grainger, J. (2020). The role of attention in word recognition: Results from OB1‐reader. Cognitive science, 44(7), e12846.
* Grainger, J., & Van Heuven, W. J. (2004). Modeling letter position coding in printed word perception.
* Engbert, R., Nuthmann, A., Richter, E. M., & Kliegl, R. (2005). SWIFT: a dynamical model of saccade generation during reading. Psychological review, 112(4), 777.
* Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with predictability norms. Behavior research methods, 50, 826-833.
* Van Heuven, W. J., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK: A new and improved word frequency database for British English. Quarterly journal of experimental psychology, 67(6), 1176-1190.