import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
SEED = 13

if __name__ == "__main__":
    #error when unzip, del
    path = "C:\\Users\\pka\\kaggle\\panda\\input"    
    #this only for my i get error when unzip files and skip it
    drop = [
        'fa7ece1587e37e28416110063cc5266a',
        'fc20aec4fd0c2f9017888dd94feb84ee'
        ]
    
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    df = df[~df.image_id.isin(drop)]
    #shuffle data
    #df = df.sample(frac = 1).reset_index(drop=True)    
    #folds    
    split = StratifiedKFold(5, random_state=SEED, shuffle = True)
    idx_split = list(split.split(df, df.isup_grade))
    zeros = np.zeros(len(df))
    for i in range(5):
        zeros[idx_split[i][1]] = i #array([0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1.])
    df['fold'] = zeros
    print(df.fold.value_counts())
    df.to_csv(os.path.join(path, 'train_folds.csv'), index = False)