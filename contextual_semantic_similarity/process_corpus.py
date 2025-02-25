import pandas as pd
import numpy as np
import rdata

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
            ia_new.append(text_words[i])
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
        text_word_ids = [i for i in range(len(text_words))]
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

def pre_process_eye_data(filepath, language, corpus):

    if filepath.endswith('.rda'):
        filepath = convert_rdm_to_csv(filepath)

    df = pd.read_csv(filepath)

    if corpus == 'meco':
        if 'lang' in df.columns:
            df = df[(df['lang'] == language)]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
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

    elif corpus == 'Provo':
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
        # fix tokenization
        df['Word'] = df.apply(lambda x: 'true' if x['Word'] == 'TRUE' else x['Word'], axis=1)
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
                                'IA_DWELL_TIME': 'dur'})

    return df

def add_variables(variables, df, language, corpus, frequency_filepath):

    if 'length' in variables:
        # add length and frequency
        df['length'] = [len(str(word)) for word in df['ia'].tolist()]
        df["length.log"] = np.log(df["length"])

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

    return df

def pre_process_data(eye_move_filepath, texts_filepath, frequency_filepath = '', corpus='Provo' , language='en', variables=['length']):

    # Generate a dataset with each word of each text as row.
    print('Pre-processing dataframe with texts and words from corpus...')
    if corpus == 'meco':
        # columns: trialid (the id of the text); texts (the text the word belongs to); ianum (id of the word); ia (word)
        original_df = create_original_texts_dataframe(texts_filepath, language)
        texts_df = clean_original_texts(original_df)
        words_df = create_ianum_from_original_texts(texts_df)
    elif corpus == 'Provo':
        texts_df = pd.read_csv(texts_filepath, encoding="ISO-8859-1")
        words_df = create_texts_df(texts_df)
    else:
        raise NotImplementedError(f'Corpus {corpus} not implemented.')

    print('Pre-processing dataframe with eye movements...')
    # Generate a dataset with eye-tracking dependent variables and co-variables
    eye_df = pre_process_eye_data(eye_move_filepath, language, corpus)
    eye_df = add_variables(variables, eye_df, language, corpus, frequency_filepath)

    print('Checking alignment between dataframes...')
    # check alignment between words_df and corpus_df
    words_df_dict = dict()
    for trialid, group in words_df.groupby('trialid'):
        words_df_dict[trialid] = dict()
        for ia, ianum in zip(group['ia'].tolist(), group['ianum'].tolist()):
            words_df_dict[trialid][ianum] = ia
    if corpus == 'meco': participant_col = 'uniform_id'
    elif corpus == 'Provo': participant_col = 'Participant_ID'
    for id, data in eye_df.groupby([participant_col, 'trialid']):
        for eye_ia, eye_ianum in zip(data['ia'].tolist(), data['ianum'].tolist()):
            assert eye_ianum in words_df_dict[id[1]].keys(), print(f'Word id {eye_ianum} of text {id[1]} and participant '
                                                               f'{id[0]} in eye-tracking data not in words dataframe;'
                                                               f'{group}')
            assert eye_ia == words_df_dict[id[1]][eye_ianum], print(f'Word {eye_ia} (id {eye_ianum} in text {id[1]} of participant '
                                                            f'{id[0]}) in eye-tracking dataframe does not match word '
                                                            f'of same text and id in words dataframe ({words_df_dict[id[1]][eye_ianum]}).')

    return words_df, eye_df