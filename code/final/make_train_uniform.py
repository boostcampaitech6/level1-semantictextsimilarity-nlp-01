import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from functions import text_preprocess

np.random.seed(42)
random.seed(42)

TRAIN_PATH = '../data/train.csv'
PREPROCESS_PATH = '../data/train_spellchecker.csv'
BACK_PATH = '../data/train_back.csv'
SAVE_PATH = '../data/train_uniform.csv'


def main(train_path=TRAIN_PATH, preprocessed_path=PREPROCESS_PATH, back_path=BACK_PATH, save_path=SAVE_PATH):
    # 데이터 불러오기 및 열 변경
    df_train = pd.read_csv(train_path)
    df_preprocessed = pd.read_csv(preprocessed_path, index_col=0).dropna()
    df_back = pd.read_csv(back_path, index_col=0).dropna()

    df_preprocessed = df_preprocessed[['id','source','s_sentence_1','s_sentence_2','label','binary-label']]
    df_preprocessed.columns = df_train.columns

    df_back = df_back[['id','source','b_sentence_1','b_sentence_2','label','binary-label']]
    df_back.columns = df_train.columns
    
    # Basic 전처리를 적용
    for df in [df_train, df_preprocessed, df_back]:
        for col in ['sentence_1','sentence_2']:
            df[col] = df[col].apply(text_preprocess)
    
    # 번역 품질이 떨어지는 것을 고려해서 문장 길이에 따라 -0.3 ~ -0 label panelty 부여
    max_sentence_length = max(df_back['sentence_1'].map(len) + df_back['sentence_2'].map(len))
    max_sentence_length = max(df_back['sentence_1'].map(len) + df_back['sentence_2'].map(len))
    for idx in df_back.index:
        sentence_length = len(df_back.loc[idx,'sentence_1']) + len(df_back.loc[idx,'sentence_2'])
        penalty = min(0.3, round(sentence_length/max_sentence_length,1))
        df_back.loc[idx,'label'] -= penalty
        if df_back.loc[idx,'label'] < 0:
            df_back.loc[idx,'label'] = 0.0
    
    # 0 drop 5 증강
    index0 = df_train[df_train.label==0].index
    drop_idxs = np.random.choice(index0, size=(len(index0)*3)//4,replace=False)
    copy_idxs = np.random.choice(drop_idxs, size=(20,20), replace=False).flatten()
    drop_idxs = np.setdiff1d(drop_idxs, copy_idxs)

    s1_idxs = copy_idxs[:200]
    s2_idxs = copy_idxs[200:]
    # 절반은 s1 복사, 나머지는 s2 복사
    df_train.loc[s1_idxs, 'sentence_2'] = df_train.loc[s1_idxs, 'sentence_1']
    df_train.loc[s2_idxs, 'sentence_1'] = df_train.loc[s2_idxs, 'sentence_2']
    df_train.loc[copy_idxs, 'label'] = 5.0

    df_train = df_train.drop(index=drop_idxs)

    # 평균을 기준으로 증강 및 제거
    for label in df_train.label.unique():
        df = df_train[df_train.label==label]
        df_p = df_preprocessed[df_preprocessed.label==label]
        df_b = df_back[df_back.label==label]
        size = df.shape[0]
        if size < 262//3:
            df_train = pd.concat([df_train, df_p, df_b], axis=0)
            df_train = df_train[~df_train.duplicated()]
        
        df = df_train[df_train.label==label]
        size = df.shape[0] 
        if size < 262//2:
            df_switched = df[['id','source','sentence_2','sentence_1','label','binary-label']]
            df_switched.columns = df_train.columns
            df_train = pd.concat([df_train, df_switched], axis=0)
            df_train = df_train[~df_train.duplicated()]
            
        df = df_train[df_train.label==label]
        size = df.shape[0]
        if size > 262:
            drop_size = size - 262
            drop_idxs = np.random.choice(df.index, size=drop_size, replace=False)
            df_train = df_train.drop(index=drop_idxs)
    
    # train에 없는 label은 df_back에서 penalty를 부여한 문장으로 추가합니다.
    train_labels = df_train.label.unique()
    for label in df_back.label.unique():
        if label not in train_labels:
            df = df_back[df_back.label==label]
            if df.shape[0] > 262:
                drop_size = df.shape[0] - 262
                drop_idxs = np.random.choice(df.index, size=drop_size, replace=False)
                df = df.drop(index=drop_idxs)
            df_train = pd.concat([df_train, df], axis=0)

    # 분포 확인 및 저장
    df_train.to_csv(save_path, index=False)
    print(f'Train File Save at <{save_path}>')
    # df_train.label.hist()
    # plt.show()
    
if __name__ == '__main__':
    main()
    pass