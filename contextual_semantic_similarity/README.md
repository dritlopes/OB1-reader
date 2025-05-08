# Saliency map for saccade planning
This branch of the OB1-reader repository contains the code to compute a saliency map proposed, and run the analysis reported, in the following paper:

* Lopes Rego, A. T., Meeter, M., & Snell, J. (2025). An information-theoretic mechanism to explain oculomotor behaviour in reading. _In prep._

Abstract:

"During reading, readers perform forward and backward eye movements through text, called saccades. Although such movements are intuitive to readers, the mechanisms underlying such behaviour is not yet fully known, particularly regarding the role of higher-order linguistic processes in guiding reading behaviour. One possibility is that readers tend to target the closest most informative word in the surrounding context when moving through text. In this study, we investigate the influence of semantics on saccade targeting, i.e., determining where to move our eyes next. Using contextualized embeddings from GPT-2 small and large, we measure pairwise semantic similarity between the fixated word and its surrounding words within a context window. We then analyse to what extent contextualized semantic similarity predicts the next fixation target, beyond word length, frequency and surprisal. In addition, we explore the mechanism through which this relation may take place by performing simulations with the OB1-reader, a model of eye-movement control in reading. For this end, we develop a saliency map based on semantic similarity and positional distance between the fixated word and the surrounding word, whereby the closer and less similar to the fixated word, the more likely the surrounding word is to attract attention and thus be the next fixation target. Furthermore, saliency of words in upcoming positions is also modulated by predictability and orthographic input from the parafovea to account for the uncertainty about the identity of upcoming, not yet recognized words. "

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


3. Run ```compute_semantic_similarity.py``` to obtain semantic similarity values. In the main function, you should specify the corpus name, the model and layer(s) to compute representations, and the type of semantic similarity (with previous context or within a context window). Two files are needed here: the eye-tracking file outputted by compute_surprisal.py, and the file with each word as a row outputted by process_corpus.py. The script outputs the following files:
    * File with each word as a row, with semantic similarity added as a column, if similarity with previous context. If similarity with context window, each row has the pairwise similarity with each word and each context word within defined context window.
    * File with eye-tracking data and semantic similarity values.


4. Run ```compute_saliency.py``` to obtain predicted next fixation position for each fixated word by each participant, based on saliency mapping. In the main function, specify the name of the corpus, and the model and layer used to compute surprisal and semantic similarity. Two files are needed here: the eye-tracking file outputted by compute_semantic_similarity.py, and the file with each word as a row outputted by process_corpus.py. The script outputs a file with each fixated word as a row, as well as the position of the next fixation, saliency type, and predicted position of next fixation according to saliency map.


5. Run ```analysis.py``` to evaluate accuracy and RMSE of predicted next fixation position according to saliency map. 
6. Run ```stats_analysis``` to reproduce correlational analysis of semantic similarity and eye movement variables. 
7. Run ```run.sh``` to run steps 1 to 5 on the command line if using gpu.