# This is literally someone else's implementation. 
# I am just using this to what an implementation would look like.
# https://www.kaggle.com/code/werwar23414141231/lstm-with-peephole/notebook

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# pip install -U imbalanced-learn
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import sys
args = sys.argv
for dirname, _, filenames in os.walk('../kaggle_file/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('../kaggle_file/train_input/input_2023_w01.csv')
df.describe()

########################################################################################
#################### END NOTEBOOK ######################################################
########################################################################################

if args.__contains__("-g"):
    plt.rcParams['font.size'] = 4
    fig, ax = plt.subplots(5, 5, figsize=(9, 4))
    plt.subplots_adjust(wspace=1, hspace=1)

    row = 0
    col = 0
    for value in df.columns:
        sns.histplot(df[value].values, ax=ax[row][col])
        ax[row][col].set_title(value)
        ax[row][col].set_xlim([min(df[value].values), max(df[value].values)])
        col += 1
        if col > 4:
            col = 0
            row += 1
    sns.histplot(df['player_weight'].values, ax=ax[0][0])
    ax[0][0].set_title('Player Weight')
    ax[0][0].set_xlim([min(df['player_weight'].values), max(df['player_weight'].values)])

    sns.histplot(df['player_height'].values, ax=ax[1][0], color='b')
    ax[1][0].set_title('Player Height)')
    ax[1][0].set_xlim([min(df['player_height'].values), max(df['player_height'].values)])

    sns.histplot(df['game_id'].values, ax=ax[2][0], color='b')
    ax[2][0].set_title('Game ID\'s')
    ax[2][0].set_xlim([min(df['game_id'].values) + 299, min(df['game_id'].values)+313])

    print(df)

    plt.show()


