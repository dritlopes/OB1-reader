import pandas as pd
import numpy as np
import rdata
import spacy
from sacremoses import corpus


class WordData:

    def __init__(self,
                 corpus:str,
                 filepath:str):
        self.corpus = corpus
        self.filepath = filepath
        self.data = None

    def _create_texts_provo_df(self):

        """
        Create dataframe where each text word is row. Columns: trialid (the id of the text); texts (the text the word belongs to); ianum (id of the word); ia (word))
        :return: words_df
        """

        data = pd.read_csv(self.filepath, encoding="ISO-8859-1")

        data['Word_Number'] = data.apply(
            lambda x: int(x["Word_Number"]) - 1 if (int(x["Word_Number"]) > 44) & (int(x["Text_ID"]) == 3) else int(
                x["Word_Number"]), axis=1)
        data['Word_Number'] = data.apply(
            lambda x: int(x["Word_Number"]) - 1 if (int(x["Word_Number"]) > 18) & (int(x["Text_ID"]) == 13) else int(
                x["Word_Number"]), axis=1)

        trialids, texts, words, word_ids = [], [], [], []

        for i, text_info in data.groupby('Text_ID'):

            trialid = int(i) - 1

            text = text_info['Text'].tolist()[0]
            # fix errors in texts in raw data
            text = text.replace(' Ñ', '')
            text = text.replace('Õ', "'")

            text_words = text.split()
            text_words = [word.replace('"', '') for word in text_words]

            text_word_ids = [i for i in range(len(text_words))]

            trialids.extend([trialid for i in range(len(text_words))])
            texts.extend([text for i in range(len(text_words))])
            words.extend(text_words)
            word_ids.extend(text_word_ids)

        words_df = pd.DataFrame(data={'trialid': trialids,
                                      'texts': texts,
                                      'ianum': word_ids,
                                      'ia': words})
        return words_df

    def _create_texts_meco_df(self):

        """
        Create dataframe where each text word is row. Columns: trialid (the id of the text); texts (the text the word belongs to); ianum (id of the word); ia (word))
        :return: words_df
        """

        data = pd.read_csv(self.filepath)
        data.drop(['Unnamed: 13', 'Unnamed: 14'], axis=1, inplace=True)
        data.columns = ['lang', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        # only English texts
        lan_filter = (data['lang'] == 'English')
        lan_texts_df = data.loc[lan_filter]
        # re-structure data so that each text becomes a row
        trialid_raw_df = lan_texts_df.stack().astype(str).reset_index(level=1)
        trialid_raw_df.rename(columns={'level_1': 'trialid', 0: 'text'}, inplace=True)
        trialid_raw_df = trialid_raw_df.reset_index(drop=False)
        trialid_raw_df.drop([0], inplace=True)
        trialid_raw_df.drop(['index'], axis=1, inplace=True)

        # do some cleaning on each text
        trialid_cleaning_df = trialid_raw_df.copy()
        # replace with "space" the "\\n" at the beginning of a word
        trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace(" \\n", " ", regex=False)
        # replace with "space" the "\\n" between words as "word\\nword"
        trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace("\\n", " ", regex=False)
        # when "word-word" add a space after first word, then the words would be separated equally
        trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace("-", "- ", regex=False)
        # replace with a empty string all the quotation marks
        trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace('"', '', regex=False)

        # create dataframe with each row being a word
        trialid, text, ia_new, ianum_new = [], [], [], []
        # to interact between a list of rows
        for _, row in trialid_cleaning_df.iterrows():
            # transform the text into a list of words
            text_words = row['text'].split()
            # transform text_words into a dataframe
            for i in range(0, len(text_words)):
                trialid.append(row['trialid']-1)
                # print(trialid)
                text.append(row['text'])
                # print(text)
                ianum_new.append(i)
                # print(ianum_new)
                ia_new.append(text_words[i])
                # print(ia_new)
        # adding it to dataframe
        words_df = pd.DataFrame({'trialid': trialid,
                                 'texts': text,
                                 'ianum': ianum_new,
                                 'ia': ia_new})
        return words_df

    def create_texts_df(self):

        if self.corpus == 'Provo':
            dataframe = self._create_texts_provo_df()
        elif self.corpus == 'meco':
            dataframe = self._create_texts_meco_df()
        else:
            raise Exception(f'Corpus {self.corpus} not supported.')

        self.data = dataframe

        return self.data

class ProvoData:

    def __init__(self, filepath:str):
        self.filepath = filepath
        self.data = None

    def pre_process_data(self):

        df = pd.read_csv(self.filepath, encoding="ISO-8859-1")

        # select columns
        df = df[['Participant_ID', 'Text_ID', 'Word', 'Word_Number', 'IA_FIRST_FIXATION_DURATION',
                 'IA_FIRST_RUN_DWELL_TIME', 'IA_DWELL_TIME', 'IA_SKIP', 'IA_REGRESSION_IN', 'IA_REGRESSION_OUT']]

        # drop nan values
        df.dropna(subset=['Text_ID', 'Word', 'Word_Number'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # align indexing with words_df (which starts at 0, not at 1)
        df['Text_ID'] = df['Text_ID'].apply(lambda x: int(x) - 1)
        df['Word_Number'] = df['Word_Number'].apply(lambda x: int(x) - 1)

        # fix error in ianum sequence
        df['Word_Number'] = df.apply(
            lambda x: x['Word_Number'] - 1 if (x['Text_ID'] == 2) & (x['Word_Number'] >= 45) else x['Word_Number'],
            axis=1)
        df['Word_Number'] = df.apply(
            lambda x: x['Word_Number'] - 1 if (x['Text_ID'] == 12) & (x['Word_Number'] >= 19) else x['Word_Number'],
            axis=1)
        df['Word_Number'] = df.apply(
            lambda x: 50 if (x['Text_ID'] == 17) & (x['Word_Number'] >= 2) & (x['Word'] == 'evolution') else x[
                'Word_Number'],
            axis=1)

        # reorder rows
        df.sort_values(by=['Participant_ID','Text_ID','Word_Number'], inplace=True)

        # fix tokenization to align with words_df
        df['Word'] = df.apply(lambda x: 'true' if x['Word'] == 'TRUE' else x['Word'], axis=1)
        df["Word"] = df["Word"].str.replace('"', '')
        df['Word'] = df.apply(lambda x: x['Word'].replace('?',"'") if ('?' in x['Word']) else x['Word'], axis=1)
        df['Word'] = df.apply(lambda x: '90%' if (x['Word'] == '0.9') & (x['Word_Number'] == 44) else x['Word'], axis=1)
        # words missing full stop
        miss_full_stop = []
        for i, rows in df.groupby(['Participant_ID','Text_ID']):
            last_word = rows['Word'].tolist()[-1]
            last_word_id = rows['Word_Number'].tolist()[-1]
            if '.' not in last_word[-1]:
                if i[1] != 54 and last_word_id != 59:
                    miss_full_stop.append((i[0],i[1],last_word_id))
        df["Word"] = df.apply(lambda x: x['Word'] + '.' if (x['Participant_ID'], x['Text_ID'], x['Word_Number']) in miss_full_stop else x['Word'], axis=1)

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
        self.data = df
        return self.data

    def pre_process_fixation_data(self):

        df = pd.read_csv(self.filepath, encoding="ISO-8859-1")

        # select columns
        df = df[['RECORDING_SESSION_LABEL', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_INTEREST_AREA_INDEX',
                 'CURRENT_FIX_INTEREST_AREA_LABEL',
                 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME', 'NEXT_FIX_INTEREST_AREA_INDEX', 'NEXT_FIX_INTEREST_AREA_LABEL',
                 'PREVIOUS_FIX_INTEREST_AREA_INDEX', 'PREVIOUS_FIX_INTEREST_AREA_LABEL', 'TRIAL_LABEL']]

        # rename columns
        df = df.rename(columns={'CURRENT_FIX_INDEX': 'fixid',
                                'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'dur',
                                'RECORDING_SESSION_LABEL': 'participant_id',
                                'CURRENT_FIX_INTEREST_AREA_INDEX': 'ianum',
                                'CURRENT_FIX_INTEREST_AREA_LABEL': 'ia',
                                'NEXT_FIX_INTEREST_AREA_INDEX': 'next.ianum',
                                'NEXT_FIX_INTEREST_AREA_LABEL': 'next.ia',
                                'PREVIOUS_FIX_INTEREST_AREA_LABEL': 'previous.ia',
                                'PREVIOUS_FIX_INTEREST_AREA_INDEX': 'previous.ianum'})

        # filter data of participants to only contain the same participants of Provo_Corpus-Eyetracking_Data.csv
        to_include = [id for id in df['participant_id'].unique().tolist() if 'a' not in id and id != '80']
        df = df[df['participant_id'].isin(to_include)]

        # strip trailing spaces from words
        df['ia'] = df['ia'].apply(lambda x: x.strip())

        # fix character error
        df['ia'] = df['ia'].apply(lambda x: x.replace('Õ', "'"))
        df['ia'] = df['ia'].apply(lambda x: x.replace(' Ñ', ''))
        df['ia'] = df['ia'].apply(lambda x: x.replace('Ñ', '.'))
        df['ia'] = df['ia'].apply(lambda x: '.' if x == 'livres--a' else x)
        df['ia'] = df['ia'].apply(lambda x: '.' if x == 'profession--writing.' else x)

        # drop rows with empty cell ('.')
        df['ianum'] = df['ianum'].apply(lambda x: np.nan if x == '.' else x)
        df['ia'] = df['ia'].apply(lambda x: np.nan if x == '.' else x)
        df.dropna(subset=['ia', 'ianum'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # replace '.' by empty string in previous and next ianum
        df['next.ianum'] = df['next.ianum'].apply(lambda x: np.nan if x == '.' else x)
        df['previous.ianum'] = df['previous.ianum'].apply(lambda x: np.nan if x == '.' else x)

        # compute outgoing saccade distance in words
        distances = []
        for word_id, next_word_id in zip(df['ianum'].tolist(), df['next.ianum'].tolist()):
            if not pd.isna(next_word_id):
                distances.append(int(next_word_id) - int(word_id))
            else:
                distances.append(np.nan)
        df['next_saccade_distance'] = distances

        self.data = df

        return self.data

class MecoData:
    def __init__(self, filepath:str):
        self.filepath = filepath
        self.data = None

    @staticmethod
    def _convert_rdm_to_csv(original_filepath):

        converted = rdata.read_rda(original_filepath)
        converted_key = list(converted.keys())[0]
        df = pd.DataFrame(converted[converted_key])
        filepath = original_filepath.replace('rda', 'csv')
        df.to_csv(filepath)

        return filepath

    def pre_process_data(self):

        # convert fixation report to csv
        if self.filepath.endswith('.rda'):
            self.filepath = self._convert_rdm_to_csv(self.filepath)

        df = pd.read_csv(self.filepath)

        # filter out non-english data
        if 'lang' in df.columns:
            df = df[(df['lang'] == 'en')]

        # removed unnamed columns if existent
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        #select columns
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

        self.data = df
        return self.data

    def pre_process_fixation_data(self):

        df = pd.read_csv(self.filepath)

        # filter out non-english data
        if 'lang' in df.columns:
            df = df[(df['lang'] == 'en')]

        # make sure outliers are excluded (out = fixation outside the area of the text)
        df = df[df['type']=="in"]
        df = df[['uniform_id','trialid', 'dur', 'letternum', 'letter', 'ia', 'ianum', 'ia.reg.in', 'ia.reg.out', 'ia.firstskip', 'ia.refix']]

        # trialid should start at 0, not at 1
        df['trialid'] = df['trialid'].apply(lambda x: int(x) - 1)
        # word id should start at 0 not at 1
        df['ianum'] = df['ianum'].apply(lambda x: int(x) - 1)

        # fix tokenization to align with words dataframe used to compute surprisal and embeddings
        df["ia"] = df["ia"].apply(lambda x: str(x).replace('"', ''))

        # compute outgoing saccade distance in words and in letters
        distances, let_distances = [], []
        for id, fixations in df.groupby(['uniform_id','trialid']):
            ianums = fixations['ianum'].tolist()
            letternums = fixations['letternum'].tolist()
            for i, ianum in enumerate(ianums):
                # if not last fixation, register the number of words between this and the next fixation
                if i + 1 < len(ianums):
                    distances.append(ianums[i+1] - ianum)
                    let_distances.append(letternums[i+1] - letternums[i])
                # if last fixation, no sacc.out distance
                else:
                    distances.append(None)
                    let_distances.append(None)
        df['next_saccade_distance'] = distances
        df['next_saccade_letter_distance'] = let_distances

        # rename columns to match columns from word_df and evaluation
        df = df.rename(columns={'ia.reg.in': 'reg_in',
                                'ia.reg.out': 'reg_out',
                                'ia.firstskip': 'first_skip',
                                'ia.refix': 'refix',
                                'uniform_id': 'participant_id'})

        self.data = df
        return self.data

def add_trial_ids_to_provo(fixation_df, words_df):

    """
    Add trial ids to Provo fixation dataframe.
    :param fixation_df: dataframe with fixation data
    :param words_df: dataframe with words data
    :return: fixation dataframe with trial ids added
    """

    if 'trialid' not in fixation_df.columns:

        text_ids = []
        text_words = [set(group['ia'].tolist()) for i, group in words_df.groupby('trialid')]

        for row in fixation_df.itertuples():

            word_loc = words_df.index[(words_df['ianum'] == int(row.ianum)) & (words_df['ia'] == row.ia)].tolist()

            if word_loc and len(word_loc) == 1:
                trialid = words_df.iloc[word_loc]['trialid'].tolist()[0]
                text_ids.append(int(trialid))

            # e.g. in case the same ianum-ia combination appears more than once in words_df (ambiguous as to which text each belongs to)
            # find the text with the most overlap with words in this trial label in fixation report.
            else:
                trial_rows = fixation_df[
                    (fixation_df['participant_id'] == row.participant_id) & (
                            fixation_df['TRIAL_LABEL'] == row.TRIAL_LABEL)]
                trial_words = set(trial_rows['ia'].unique())
                overlap = []
                for words in text_words:
                    overlap.append(len(words.intersection(trial_words)))
                trialid = overlap.index(max(overlap)) + 1
                text_ids.append(trialid)

        fixation_df['trialid'] = text_ids

        # change ianums from text 55 to match words_df (because of error in tokens of fixation report: livre--as)
        fixation_df['ianum'] = fixation_df.apply(
            lambda x: int(x['ianum']) + 1 if (x['trialid'] == 55) & (int(x['ianum']) > 9) else int(x['ianum']), axis=1)
        # change ianums from text 36 to match words_df (because of error in tokens of fixation report: Ñ)
        fixation_df['ianum'] = fixation_df.apply(
            lambda x: int(x['ianum']) - 1 if (x['trialid'] == 36) & (int(x['ianum']) > 24) else int(x['ianum']), axis=1)

        fixation_df = fixation_df.drop(columns=['TRIAL_LABEL'])
        fixation_df.sort_values(by=['participant_id', 'trialid', 'fixid'], inplace=True)

    return fixation_df

def add_variables(variables:list[str], df:pd.DataFrame, corpus_name:str, frequency_filepath:str):

    if 'length' in variables:
        # add length and frequency
        df['length'] = [len(str(word)) for word in df['ia'].tolist()]
        df["length_log"] = np.log(df["length"])

    if 'frequency' in variables and frequency_filepath:

        if corpus_name == 'meco': # we use frequency file from meco corpus
            freq_col_name = 'zipf_freq'
            word_col_name = 'ia_clean'
            frequency_df = pd.read_csv(frequency_filepath, usecols=[freq_col_name, word_col_name])
            if 'lang' in frequency_df.columns:
                frequency_df = frequency_df[frequency_df['lang'] == 'english']
        elif corpus_name == 'Provo': # we use SUBTLEX-UK
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

def check_alignment(words_df: pd.DataFrame, eye_df: pd.DataFrame):

    """
    Check alignment between word and fixation dataframes (whether word ids match).
    :param words_df: words dataframe
    :param eye_df: fixation dataframe
    """

    # create dict with text id and word id as keys and word form as value
    words_df_dict = dict()
    for trialid, group in words_df.groupby('trialid'):
        words_df_dict[trialid] = dict()
        for ia, ianum in zip(group['ia'].tolist(), group['ianum'].tolist()):
            words_df_dict[trialid][ianum] = ia

    # for each word if and word in eye-movement dataframe, check if it's the same in word dataframe
    for id, data in eye_df.groupby(['participant_id', 'trialid']):
        for eye_ia, eye_ianum in zip(data['ia'].tolist(), data['ianum'].tolist()):
            # in case word_id-word combination from eye-movement dataframe does not exist in words dataframe
            assert eye_ianum in words_df_dict[id[1]].keys(), print(
                f'Word id {eye_ianum} of text {id[1]} and participant '
                f'{id[0]} in eye-tracking data not in words dataframe;'
                f'{group}')
            # in case word from eye-movement dataframe does not match word with same id in words dataframe
            assert eye_ia == words_df_dict[id[1]][eye_ianum], print(
                f'Word {eye_ia} (id {eye_ianum} in text {id[1]} of participant '
                f'{id[0]}) in eye-tracking dataframe does not match word '
                f'of same text and id in words dataframe ({words_df_dict[id[1]][eye_ianum]}).')

def main():

    texts_filepath = 'data/raw/meco/supp_texts.csv'  # 'data/raw/Provo/Provo_Corpus-Predictability_Norms.csv' # 'data/raw/meco/supp_texts.csv'
    eye_move_filepath = 'data/raw/meco/joint_fix_trimmed.csv'  # 'data/raw/Provo/Provo_Corpus-Eyetracking_Data.csv' # data/raw/Provo/Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv # 'data/raw/meco/joint_fix_trimmed.rda' # data/raw/meco/joint_data_trimmed.rda
    frequency_filepath = 'data/raw/meco/wordlist_meco.csv'  # 'data/raw/Provo/SUBTLEX_UK.txt' # 'data/raw/meco/wordlist_meco.csv'
    corpus_name = 'meco'  # 'Provo' 'meco' # TODO run pre-process fixation report from Provo
    words_filepath = f'data/processed/{corpus_name}/words_en_df.csv'
    processed_eye_move_filepath = f'data/processed/{corpus_name}/fixation_report_en_df.csv'  # corpus_{language}_df.csv if word-based data; fixation_report_{language}_df if fixation report

    # Word Data
    print('Pre-processing dataframe with texts...')
    # words_data = WordData(corpus_name, texts_filepath).create_texts_df()
    # words_data.to_csv(words_filepath, index=False)
    words_data = pd.read_csv(words_filepath, index_col=0)

    # Eye-movement Data
    print('Pre-processing dataframe with eye movements...')
    if corpus_name == 'meco':
        eye = MecoData(eye_move_filepath)
        if "joint_fix_trimmed" in eye_move_filepath:
            eye_data = eye.pre_process_fixation_data()
        elif "joint_data_trimmed" in eye_move_filepath:
            eye_data = eye.pre_process_data()
    elif corpus_name == 'Provo':
        eye = ProvoData(eye_move_filepath)
        if "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report" in eye_move_filepath:
            eye_data = eye.pre_process_fixation_data()
            eye_data = add_trial_ids_to_provo(eye_data, words_data)
        elif "Provo_Corpus-Eyetracking_Data" in eye_move_filepath:
            eye_data = eye.pre_process_data()
    else:
        raise Exception(f'Corpus {corpus_name} not implemented.')
    eye_data = add_variables(['length', 'frequency', 'pos_tag'], eye_data, corpus_name, frequency_filepath)
    eye_data.to_csv(processed_eye_move_filepath, index=False)

    # Check alignment
    check_alignment(words_data, eye_data)

if __name__ == '__main__':
    main()