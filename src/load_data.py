import json
from copy import deepcopy
import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm

def preprocess_data(folder_name):
    data = (json.load(open(folder_name, "rb+")))
    # data_dict = dictify(data)
    dataframe = pd.DataFrame.from_dict(data)
    stats = unroll_stats(dataframe['histogram'])
    dataframe = dataframe.drop(columns = ['histogram'])
    processed_dataframe = pd.concat([dataframe, stats], axis=1, sort=False)
    return processed_dataframe

def get_char_stats(hist_df):
    char_codes = [32, 101, 105, 108, 110, 114, 116, 115, 97, 117, 111, 100, 121, 99, 104, 103, 109, 112, 98, 107, 118,
                  119, 102, 122, 120, 113, 106]
    for char_code in tqdm(char_codes):
        hitCount = []
        missCount = []
        timeToType = []
        for i in range(len(hist_df['histogram'])):
            char_codes_present = [x['charCode'] for x in hist_df['histogram'][i]]
            if char_code in char_codes_present:
                ind = char_codes_present.index(char_code)
            else:
                ind = None
            if not ind is None:
                hitCount.append(int(hist_df['histogram'][i][ind]['hitCount']))
                missCount.append(int(hist_df['histogram'][i][ind]['missCount']))
                timeToType.append(int(hist_df['histogram'][i][ind]['timeToType']))
            else:
                hitCount.append(np.nan)
                missCount.append(np.nan)
                timeToType.append(np.nan)
        hist_df[f'\'{chr(char_code)}\'_hitCount'] = np.array(hitCount)
        hist_df[f'\'{chr(char_code)}\'_missCount'] = np.array(missCount)
        hist_df[f'\'{chr(char_code)}\'_timeToType'] = np.array(timeToType)
    # delete the first column
    hist_df = hist_df.drop(columns = ['histogram'])
    return hist_df

def unroll_stats(histogram):
    hist_df = pd.DataFrame.from_dict(histogram)
    hist_df = get_char_stats(hist_df)
    return hist_df




