# Predicting saccade targeting in reading
This branch of the OB1-reader repository contains the code to train and test a model for predicting saccade targeting in reading, proposed in the following paper:

* Lopes Rego, A. T., Snell, J., & Meeter, M. (2025). What determines where readers fixate next? Leveraging NLP to investigate human cognition. _In prep._

Abstract:

"During reading, readers perform forward and backward eye movements through text, called saccades. Although this behaviour is intuitive to readers, the mechanisms underlying it are not yet fully known, particularly regarding the role of higher-order linguistic processes in guiding eye-movement behaviour in naturalistic reading. One possibility is that readers tend to target the closest and most informative word in the surrounding context when moving through text. Current models of eye movement simulation in reading either limit the role of high-order linguistic information or lack explainability and cognitive plausibility. In this study, we investigate the influence of linguistic information on saccade targeting, i.e. determining where to move our eyes next, by predicting the location of the next fixation based on a limited processing window, more similarly to the amount of information humans readers can presumably process in parallel within the visual field at each fixation. Our preliminary results suggest that, while word length and frequency are important factors for determining the target of forward saccades, the contextualized meaning of the previous sequence, as well as whether the context word had been fixated before and the distance of the previous saccade, are important factors for predicting backward saccades."

### How to run the code

1. Run ```process_corpus.py``` to obtain the pre-processed corpus files to be further used in the next computation steps. Make sure you have the corpus files, and at the path being specified in the main function of the script. Currently, we support the English part of MECO, and Provo. 

    The following files are needed:
   * MECO: **supp_texts.txt**, which contains the texts read by the participants; **joint_fix_trimmed.rda**, which contains the fixation report; and **wordlist_meco.csv**, which contains the word frequency values. All files can be obtained at https://osf.io/3527a/.
   * Provo: **Provo_Corpus-Predictability_Norms.csv** which contains the texts read by the participants; **Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv**, which contains the fixation report; and **SUBTLEX_UK**, which contains the word frequency values. The first two files can be found at https://osf.io/sjefs/, and the last file can be found at https://osf.io/zq49t/.
    
   The following files get outputted:
   * File with each word as row (for further computation of surprisal and semantic_similarity), and file with processed eye-tracking data.


2. Run ```compute_surprisal.py``` to obtain the eye-tracking data including surprisal values for each word. In the main function, you can specify which language model (gpt2 or llama) and which corpus (provo or meco) should be used. The two files output from process_corpus.py are needed here. The script outputs the following files:
    * File with each word as row, with surprisal (and entropy) added as a column.
    * File with eye-tracking data with surprisal values (and entropy) added as a column.


3. Run ```compute_embeddings.py``` to obtain contextualized embeddings. In the main function, you should specify the corpus name, the model and layer(s) to compute representations. Two files are needed here: the eye-tracking file outputted by compute_surprisal.py, and the file with each word as a row outputted by process_corpus.py. The script outputs the following files:
    * A file for each corpus text containing the pytorch tensor embeddings for each word position in the text.
    * A file containing the map between word position in the corpus text and token position in the language model input for further alignment.


4. Run ```train_classifier.py``` to train and test classifier, and to run feature ablation. In the main function, provide the following arguments:
   * ```eye_data_filepath``` = the filepath to the corpus data (output in step 1 - and 2 if using surprisal values); We provide this file for the MECO corpus under 'data/processed/meco/gpt2' for convenience.
   * ```word_data_filepath``` = the filepath to the words data (output in step 1). We provide this file for the MECO corpus under 'data/processed/meco' for convenience.
   * ```opt_dir``` = the path to the directory where to store the output of the training and testing.
   * ```compute_tensors``` = whether to compute tensors which form the input vectors. They may be computed and stored already, in which case set this boolean to False.
   * ```pre_process``` = whether to pre-process the corpus data file, e.g. z-normalize the variables. If data is already pre_processed (we provide the pre-processed file whose name finishes with "cleaned"), set this boolean to False.
   * ```norm_method``` = normalization method. Either 'z-score' or 'max-min'.
   * ```baselines``` = define which baselines to compare the trained model with (default = 'next_word,random')
   * ```features``` = which features to use in the input. Full model is 'length,surprisal,frequency,has_been_fixated,embedding,previous_sacc_distance,previous_fix_duration'
   * ```vectors_dir``` = directory in which to save the computed vectors.
   * ```n_context_words``` = how many word positions to consider for targeting. Default is 7.
   * ```params_dataloader``` = the parameters for the Pytorch DataLoader. Default is batch_size = 32, shuffle = True, and num_workers = 6.
   * ```params_classifier``` = the parameters for the model Classifier. Default is hidden_nodes = 128, output_nodes = n_context_words, and learning_rate = .0001.
   * ```epochs``` = the number of epochs for training. Default is 10.
   * ```seed``` = the seed for reproducibility of random steps. Default is 42.
   * ```do_training``` = set to True to train model.
   * ```do_feature_ablation``` = set to True to run feature ablation.


5. Run ```run.sh``` to train classifier on the command line if using gpu.