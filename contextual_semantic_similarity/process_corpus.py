import pandas as pd
import numpy as np
import rdata
import spacy

def create_original_texts_dataframe(file_path, language):

    texts_df = pd.read_csv(file_path, sep=',')
    texts_df.drop(['Unnamed: 13', 'Unnamed: 14'], axis=1, inplace=True)
    texts_df.columns = ['lang', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    filter = ''
    if language == 'en': filter = 'English'
    elif language == 'du': filter = 'Dutch'
    lan_filter = (texts_df['lang'] == filter)
    lan_texts_df = texts_df.loc[lan_filter]

    trialid_raw_df = lan_texts_df.stack().astype(str).reset_index(level=1)
    trialid_raw_df.rename(columns={'level_1':'trialid', 0:'text'}, inplace=True)
    trialid_raw_df = trialid_raw_df.reset_index(drop=False)
    trialid_raw_df.drop([0], inplace=True)
    trialid_raw_df.drop(['index'], axis=1, inplace=True)

    return trialid_raw_df

def clean_original_texts(trialid_raw_df: pd) -> pd.DataFrame:

    trialid_cleaning_df = trialid_raw_df.copy()

    # replace with "space" the "\\n" at the beginning of a word
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].apply(lambda x: str(x).replace(" \\n", " "))
    # replace with "space" the "\\n" between words as "word\\nword"
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].apply(lambda x: str(x).replace("\\n", " "))
    # when "word-word" add a space after first word, then the words would be separated equally
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].apply(lambda x: str(x).replace("-", "- "))
    # replace all the quotation marks with an empty string
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].apply(lambda x: str(x).replace('"', ''))
    # replace \n in the\ndoors
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].apply(lambda x: str(x).replace("the\ndoors", "the doors"))

    return trialid_cleaning_df

def create_ianum_from_original_texts(texts_df: pd):

    trialid = []
    text = []
    ia_new = []
    ianum_new = []

    for _, row in texts_df.iterrows():
        text_words = row['text'].split()
        for i in range(0, len(text_words)):
            trialid.append((row['trialid'])-1) # to start at 0
            text.append(row['text'])
            ianum_new.append(i)
            word = text_words[i]
            word = word.replace('"','')
            ia_new.append(word)
    words_df = pd.DataFrame({'id': [i for i in range(len(trialid))],
                            'trialid': trialid,
                            'texts': text,
                            'ianum': ianum_new,
                            'ia': ia_new})

    return words_df

def create_texts_df(data):

    trialids, texts, words, word_ids = [], [], [], []

    for i, text_info in data.groupby('Text_ID'):

        trialid = int(i) - 1
        text = text_info['Text'].tolist()[0]
        # fix errors in texts in raw data
        if int(i) == 36:
            text = text.replace(' Ñ', '')
        if int(i) == 27:
            text = text.replace('Õ', "'")
        if int(i) == 45:
            text = text.replace('Õ', "'")
        if int(i) == 54:
            text = text.replace('Õ', "'")

        text_words = text.split()
        text_words = [word.replace('"', '') for word in text_words]
        text_word_ids = [i for i in range(len(text_words))]
        print(text_words)
        words.extend(text_words)
        word_ids.extend(text_word_ids)
        texts.extend([text for i in range(len(text_words))])
        trialids.extend([trialid for i in range(len(text_words))])

    data = pd.DataFrame(data={'id': [i for i in range(len(trialids))],
                              'trialid': trialids,
                              'texts': texts,
                              'ianum': word_ids,
                              'ia': words})
    return data

def convert_rdm_to_csv(original_filepath):

    converted = rdata.read_rda(original_filepath)
    converted_key = list(converted.keys())[0]
    df = pd.DataFrame(converted[converted_key])
    filepath = original_filepath.replace('rda', 'csv')
    df.to_csv(filepath)

    return filepath

def pre_process_eye_data(filepath, language):

    if filepath.endswith('.rda'):
        filepath = convert_rdm_to_csv(filepath)

    encoding = 'utf-8'
    if 'Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report' in filepath:
        encoding="ISO-8859-1"

    df = pd.read_csv(filepath, encoding=encoding)

    # Word-based data from MECO
    if "joint_data_trimmed" in filepath:
        if 'lang' in df.columns:
            df = df[(df['lang'] == language)]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # select columns
        df = df[['uniform_id', 'trialid', 'ia', 'ianum', 'reread', 'dur', 'reg.in', 'reg.out', 'skip', 'singlefix', 'firstrun.dur', 'firstfix.dur']]
        # drop rows with empty word
        df['ia'] = df['ia'].replace(' ', np.nan)
        df = df.dropna(subset=['ia'])
        df = df.reset_index(drop=True)
        # trialid should start at 0, not at 1
        df['trialid'] = df['trialid'].apply(lambda x: int(x) - 1)
        # re-index words (bcs of dropping rows with empty word)
        df['ianum'] = df['ianum'].apply(lambda x: int(x) - 1)
        # fix error in ianum sequence
        df['ianum'] = df.apply(
            lambda x: x['ianum'] - 1 if (x['ianum'] >= 149)
                                        & (x['trialid'] == 2)
                                        & (x['uniform_id'] in [f'en_{str(p)}' for p in [101,102,103,3,6,72,74,76,78,79,82,83,84,85,86,87,88,89,90,91,93,94,95,97,98,99]])
                                        else x['ianum'], axis=1)
        # fix tokenization to align with words_df
        df["ia"] = df["ia"].str.replace('"', '')
        # rename columns to match columns from word_df and evaluation
        df = df.rename(columns={'firstrun.dur': 'gaze_dur',
                                'firstfix.dur': 'first_fix_dur',
                                'uniform_id': 'participant_id',
                                'reg.in': 'reg_in',
                                'reg.out': 'reg_out'})


    # Word-based data from Provo
    elif 'Provo_Corpus-Eyetracking_Data' in filepath:
        # select columns
        df = df[['Participant_ID', 'Text_ID', 'Word', 'Word_Number', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_DWELL_TIME', 'IA_SKIP', 'IA_REGRESSION_IN', 'IA_REGRESSION_OUT']]
        # drop nan values
        df.dropna(subset=['Text_ID', 'Word', 'Word_Number'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # align indexing with words_df (which starts at 0, not at 1)
        df['Text_ID'] = df['Text_ID'].apply(lambda x: int(x) - 1)
        df['Word_Number'] = df['Word_Number'].apply(lambda x: int(x) - 1)
        # fix error in ianum sequence
        df['Word_Number'] = df.apply(
            lambda x: x['Word_Number']-1 if (x['Text_ID'] == 2) & (x['Word_Number'] >= 45) else x['Word_Number'], axis=1)
        df['Word_Number'] = df.apply(
            lambda x: x['Word_Number']-1 if (x['Text_ID'] == 12) & (x['Word_Number'] >= 19) else x['Word_Number'], axis=1)
        df['Word_Number'] = df.apply(
            lambda x: 50 if (x['Text_ID'] == 17) & (x['Word_Number'] >= 2) & (x['Word'] == 'evolution') else x['Word_Number'],
            axis=1)
        # fix tokenization to align with words_df
        df['Word'] = df.apply(lambda x: 'true' if x['Word'] == 'TRUE' else x['Word'], axis=1)
        df["Word"] = df["Word"].str.replace('"', '')
        map = {'nationwide.': ['nationwide', 57, 2],
               'possible.': ['possible', 56, 7],
               'carts.': ['carts', 52, 12],
               'stranded.': ['stranded', 52, 13],
               'jury.': ['jury', 52, 14],
               'evolution.': ['evolution', 50, 17],
               'temperature.': ['temperature', 50, 18],
               'worth.': ['worth', 47, 23],
               '90%': ['0.9', 44, 24],
               'process.': ['process', 47, 24],
               'October.': ['October', 45, 27],
               'pole.': ['pole', 43, 30],
               'mathematics.': ['mathematics', 42, 31],
               'car.': ['car', 42, 32],
               'altogether.': ['altogether', 40, 34],
               'world.': ['world', 38, 35],
               'react.': ['react', 39, 37],
               'made.': ['made', 45, 39],
               'decibels.': ['decibels', 45, 40],
               'tired.': ['tired', 46, 41],
               'explosion.': ['explosion', 47, 42],
               "women's": ['women?s', 27, 44],
               'vote.': ['vote', 48, 44],
               'States.': ['States', 49, 47],
               'companies.': ['companies', 53, 50],
               'all.': ['all', 55, 52],
               "bonds'": ['bonds?', 26, 53],
               "money.": ["money", 55, 53]}
        for new, condition in map.items():
            df['Word'] = df.apply(lambda x: new if (x['Word'] == condition[0])
                                                             & (x['Word_Number'] == condition[1])
                                                             & (x['Text_ID'] == condition[2]) else x['Word'], axis=1)
        # rename columns to match columns from words_df, based on which we extract LM estimates
        df = df.rename(columns={'Word': 'ia',
                                'Word_Number': 'ianum',
                                'Text_ID': 'trialid',
                                'IA_SKIP': 'skip',
                                'IA_DWELL_TIME': 'dur',
                                'IA_FIRST_FIXATION_DURATION': 'first_fix_dur',
                                'IA_FIRST_RUN_DWELL_TIME': 'gaze_dur',
                                'IA_REGRESSION_IN': 'reg_in',
                                'IA_REGRESSION_OUT': 'reg_out',
                                'Participant_ID': 'participant_id'})

    # Fixation report from MECO
    elif "joint_fix_trimmed" in filepath:
        df = df[(df['lang'] == language)]
        # make sure outliers are excluded (out = fixation outside the area of the text)
        df = df[df['type']=="in"]
        df = df[['uniform_id','trialid', 'dur', 'ia', 'ianum', 'ia.reg.in', 'ia.reg.out', 'ia.firstskip', 'ia.refix']]
        # trialid should start at 0, not at 1
        df['trialid'] = df['trialid'].apply(lambda x: int(x) - 1)
        # word id should start at 0 not at 1
        df['ianum'] = df['ianum'].apply(lambda x: int(x) - 1)
        # fix tokenization to align with words dataframe used to compute surprisal and embeddings
        df["ia"] = df["ia"].apply(lambda x: str(x).replace('"', ''))
        # compute outgoing saccade distance in words
        distances = []
        for id, fixations in df.groupby(['uniform_id','trialid']):
            ianums = fixations['ianum'].tolist()
            for i, ianum in enumerate(ianums):
                # if not last fixation, register the number of words between this and the next fixation
                if i + 1 < len(ianums):
                    distances.append(ianums[i+1] - ianum)
                # if last fixation, no sacc.out distance
                else:
                    distances.append(None)
        df['landing_target_position'] = distances
        # rename columns to match columns from word_df and evaluation
        df = df.rename(columns={'ia.reg.in': 'reg_in',
                                'ia.reg.out': 'reg_out',
                                'ia.firstskip': 'first_skip',
                                'ia.refix': 'refix',
                                'uniform_id': 'participant_id'})

    # Fixation report from Provo
    # Not using it because it does not have trialid (no way to align with words_df bcs it doesn't say from which text the words come from)
    elif "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report" in filepath:
        # select columns
        df = df[['RECORDING_SESSION_LABEL', 'CURRENT_FIX_INTEREST_AREA_INDEX', 'CURRENT_FIX_INTEREST_AREA_LABEL', 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME', 'NEXT_FIX_INTEREST_AREA_INDEX', 'NEXT_FIX_INTEREST_AREA_LABEL', 'PREVIOUS_FIX_INTEREST_AREA_INDEX', 'PREVIOUS_FIX_INTEREST_AREA_LABEL']]
        # rename columns
        df = df.rename(columns={'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'dur',
                                'RECORDING_SESSION_LABEL': 'participant.id',
                                'CURRENT_FIX_INTEREST_AREA_INDEX': 'ianum',
                                'CURRENT_FIX_INTEREST_AREA_LABEL': 'ia',
                                'NEXT_FIX_INTEREST_AREA_INDEX': 'next.ianum',
                                'NEXT_FIX_INTEREST_AREA_LABEL': 'next.ia',
                                'PREVIOUS_FIX_INTEREST_AREA_LABEL': 'previous.ia',
                                'PREVIOUS_FIX_INTEREST_AREA_INDEX': 'previous.ianum'})
        # drop rows with empty cell ('.')
        df['ianum'] = df['ianum'].replace('.', np.nan)
        df['ia'] = df['ia'].replace('.', np.nan)
        df.dropna(subset=['ia', 'ianum'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # reindex text id and word id to match words_df
        df['ianum'] = df['ianum'].apply(lambda x: int(x) - 1)
        df['next.ianum'] = df['next.ianum'].apply(lambda x: int(x) - 1 if x != '.' else x)
        df['previous.ianum'] = df['previous.ianum'].apply(lambda x: int(x) - 1 if x != '.' else x)
        # replace '.' by None
        df['next.ianum'] = df['next.ianum'].replace('.', None)
        df['previous.ianum'] = df['previous.ianum'].replace('.', None)
        # compute outgoing saccade distance in words
        distances = []
        for word_id, next_word_id in zip(df['ianum'].tolist(), df['next.ianum'].tolist()):
            if next_word_id:
                distances.append(int(next_word_id) - int(word_id))
            else:
                distances.append(None)
        df['landing_target_position'] = distances

    else:
        raise NotImplementedError('Corpus not supported.')

    return df

def add_variables(variables, df, language, corpus, frequency_filepath):

    if 'length' in variables:
        # add length and frequency
        df['length'] = [len(str(word)) for word in df['ia'].tolist()]
        df["length_log"] = np.log(df["length"])

    if 'frequency' in variables and frequency_filepath:

        if corpus == 'meco': # we use frequency file from meco corpus
            freq_col_name = 'zipf_freq'
            word_col_name = 'ia_clean'
            frequency_df = pd.read_csv(frequency_filepath, usecols=[freq_col_name, word_col_name])
            if language == 'en':
                language = 'english'
            if language == 'du':
                language = 'dutch'
            if 'lang' in frequency_df.columns:
                frequency_df = frequency_df[frequency_df['lang'] == language]
        elif corpus == 'Provo': # we use SUBTLEX-UK
            freq_col_name = 'LogFreq(Zipf)'
            word_col_name = 'Spelling'
            frequency_df = pd.read_csv(frequency_filepath, sep='\t',
                                       usecols=[freq_col_name, word_col_name],
                                       dtype={word_col_name: np.dtype(str)})
        else:
            raise NotImplementedError('Frequency resource or corpus not implemented.')

        frequency_col = []
        for word in df['ia'].tolist():
            word = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), str(word)))
            if word.isalpha():
                word = word.lower()
            if word in frequency_df[word_col_name].tolist():
                frequency_col.append(frequency_df[freq_col_name].tolist()[frequency_df[word_col_name].tolist().index(word)])
            else:
                frequency_col.append(None)
        df['frequency'] = frequency_col

    if 'pos_tag' in variables:
        pos_tag_col = []
        nlp = spacy.load("en_core_web_sm")
        for word in df['ia'].tolist():
            doc = nlp(word)
            pos_tag_col.append(doc[0].pos_)
        df['pos_tag'] = pos_tag_col

    return df

def pre_process_word_data(texts_filepath, frequency_filepath = '', corpus='meco' , language='en', variables=['length']):

    # Generate a dataset with each word of each text as row.
    print('Pre-processing dataframe with texts...')
    if corpus == 'meco':
        # columns: trialid (the id of the text); texts (the text the word belongs to); ianum (id of the word); ia (word)
        original_df = create_original_texts_dataframe(texts_filepath, language)
        texts_df = clean_original_texts(original_df)
        words_df = create_ianum_from_original_texts(texts_df)
        # words_df = pd.read_csv(f'data/processed/{corpus}/words_en_df.csv')
        words_df = add_variables(variables, words_df, language, corpus, frequency_filepath)
    elif corpus == 'Provo':
        texts_df = pd.read_csv(texts_filepath, encoding="ISO-8859-1")
        words_df = create_texts_df(texts_df)
    else:
        raise NotImplementedError(f'Corpus {corpus} not implemented.')

    return words_df

def pre_process_corpus_data(eye_move_filepath, words_df, frequency_filepath = '', corpus='meco' , language='en', variables=['length']):

    print('Pre-processing dataframe with eye movements...')
    # Generate a dataset with eye-tracking dependent variables and co-variables
    eye_df = pre_process_eye_data(eye_move_filepath, language)
    eye_df = add_variables(variables, eye_df, language, corpus, frequency_filepath)

    print('Checking alignment between dataframes...')
    # check alignment between words_df and corpus_df
    words_df_dict = dict()
    for trialid, group in words_df.groupby('trialid'):
        words_df_dict[trialid] = dict()
        for ia, ianum in zip(group['ia'].tolist(), group['ianum'].tolist()):
            words_df_dict[trialid][ianum] = ia
    for id, data in eye_df.groupby(['participant_id', 'trialid']):
        for eye_ia, eye_ianum in zip(data['ia'].tolist(), data['ianum'].tolist()):
            assert eye_ianum in words_df_dict[id[1]].keys(), print(f'Word id {eye_ianum} of text {id[1]} and participant '
                                                               f'{id[0]} in eye-tracking data not in words dataframe;'
                                                               f'{group}')
            assert eye_ia == words_df_dict[id[1]][eye_ianum], print(f'Word {eye_ia} (id {eye_ianum} in text {id[1]} of participant '
                                                            f'{id[0]}) in eye-tracking dataframe does not match word '
                                                            f'of same text and id in words dataframe ({words_df_dict[id[1]][eye_ianum]}).')
    return eye_df

def main():

    language = 'en'
    texts_filepath = 'data/raw/meco/supp_texts.csv'  # 'data/raw/Provo/Provo_Corpus-Predictability_Norms.csv' # 'data/raw/meco/supp_texts.csv'
    eye_move_filepath = 'data/raw/meco/joint_fix_trimmed.csv'  # 'data/raw/Provo/Provo_Corpus-Eyetracking_Data.csv' # data/raw/Provo/Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv # 'data/raw/meco/joint_data_trimmed.rda' # data/raw/meco/joint_data_trimmed.csv
    frequency_filepath = 'data/raw/meco/wordlist_meco.csv'  # 'data/raw/Provo/SUBTLEX_UK.txt' # 'data/raw/meco/wordlist_meco.csv'
    corpus_name = 'meco'  # 'Provo' 'meco'

    words_filepath = f'data/processed/{corpus_name}/words_{language}_df.csv'
    processed_eye_move_filepath = f'data/processed/{corpus_name}/fixation_report_{language}_df.csv'  # corpus_{language}_df.csv if word-based data; fixation_report_{language}_df if fixation report

    words_df = pre_process_word_data(texts_filepath, frequency_filepath, corpus_name, language, variables=['length', 'frequency', 'pos_tag'])
    words_df.to_csv(words_filepath, index=False)

    eye_move_df = pre_process_corpus_data(eye_move_filepath, words_df, frequency_filepath, corpus_name,
                                                 language, variables=['length', 'frequency'])
    eye_move_df.to_csv(processed_eye_move_filepath, index=False)

if __name__ == '__main__':
    main()