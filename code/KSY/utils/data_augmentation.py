import pandas as pd
import random
from unicode import split_syllables, join_jamos
from collections import defaultdict

typos_dict = {'ㅂ' : [ 'ㅈ','ㅁ','1','2','q','ㅃ'],
              'ㅃ' : ['ㅂ'],
              'ㅉ' : ['ㅈ'],
              'ㄸ' : ['ㄷ'],
              'ㄲ' : ['ㄱ'],
             'ㅈ' : ['2','3','ㅂ','ㄷ','ㅁ','ㄴ','ㅇ','w','ㅉ'],
             'ㄷ' : ['ㅈ','3','4','ㄱ','ㄴ','ㅇ','e','ㄸ'],
             'ㄱ' : ['ㄷ','4','5','ㅅ','ㅇ','ㄹ','r','ㄲ'],
             'ㅅ' : ['ㄱ','5','6','ㅛ','ㄹ','ㅎ','t','ㅆ'],
             'ㅁ' : ['ㅂ','ㅈ','ㄴ','ㅋ','a'],
             'ㄴ' : ['ㅁ','ㅈ','ㄷ','ㅇ','ㅋ','ㅌ','s'],
             'ㅇ' : ['ㄴ','ㄷ','ㄱ','ㄹ','ㅌ','ㅊ','d'],
             'ㄹ' : ['ㅇ','ㄱ','ㅅ','ㅎ','ㅊ','ㅍ','f'],
             'ㅎ' : ['ㄹ','ㅅ','ㅛ','ㅗ','ㅍ','ㅠ','g'],
             'ㅋ' : ['ㅁ','ㄴ','ㅊ','z'],
             'ㅌ' : ['ㅋ','ㄴ','ㅇ','ㅊ','x'],
             'ㅊ' : ['ㅌ','ㅇ','ㄹ','ㅍ','c'],
             'ㅍ' : ['ㅊ','ㄹ','ㅎ','ㅠ','v'],
             'ㅛ' : ['ㅅ','6','7','ㅕ','ㅎ','ㅗ','y'],
             'ㅕ' : ['ㅛ','7','8','ㅑ','ㅗ','ㅓ','u'],
             'ㅑ' : ['ㅕ','8','9','ㅐ','ㅏ','ㅓ','i'],
             'ㅐ' : ['ㅒ','ㅑ','9','0','ㅔ','ㅣ','ㅏ','o'],
             'ㅒ' : ['ㅐ'],
             'ㅔ' : ['ㅐ','0','ㅣ','ㅖ','p'],
             'ㅖ' : ['ㅔ'],
             'ㅗ' : ['ㅎ','ㅛ','ㅕ','ㅓ','ㅠ','ㅜ','h'],
             'ㅓ' : ['ㅗ','ㅕ','ㅑ','ㅏ','ㅜ','ㅡ','j'],
             'ㅏ' : ['ㅓ','ㅑ','ㅐ','ㅣ','ㅡ','k'],
             'ㅣ' : ['ㅏ','ㅐ','ㅔ','l'],
             'ㅠ' : ['ㅍ','ㅎ','ㅗ','ㅜ','b'],
             'ㅜ' : ['ㅠ','ㅗ','ㅓ','ㅡ','n'],
             'ㅡ' : ['ㅜ','ㅓ','ㅏ','m'],
             '?' : ['/'],
             '!' : ['1'],
             ' ' : [' '],
             'ㅢ' : ['ㅢ']}

typos_defaultdict = defaultdict(lambda : [''])
typos_defaultdict.update(typos_dict)

def generate_noise(sentence:str, mod_num: int = 3) -> str:
    syllables = list(split_syllables(sentence))

    choice_idx = random.sample(range(1,len(syllables)),mod_num)
    choice_char = [syllables[choice_idx[i]] for i in range(mod_num)]

    for i in range(mod_num):
        syllables[choice_idx[i]] = random.choice(typos_defaultdict[choice_char[i]])

    return join_jamos(''.join(syllables))

def make_blank(sentence:str, mod_num: int = 3) -> str:
    sentence = list(sentence)
    n = len(sentence)

    choice_idx = random.sample(range(1,n),mod_num)

    for i in choice_idx:
        sentence.insert(i,' ')

    return "".join(sentence).replace('  ',' ')
    
def noise_augmentation(data: pd.DataFrame,target: list,op: str= "op1") -> pd.DataFrame:
    """
    노이즈 추가
        op1. 오타 추가
        op2. 공백 추가
        op3. 랜덤 유의어
    """
    data['round-label'] = data[target].apply(round)
    df_123 = data.loc[data['round-label'].isin([1, 2, 3]), :]

    if op=="op1":
        df_123['sentence_1'] = df_123['sentence_1'].apply(generate_noise)
        df_123['sentence_2'] = df_123['sentence_2'].apply(generate_noise)
    elif op=="op2":
        df_123['sentence_1'] = df_123['sentence_1'].apply(make_blank)
        df_123['sentence_2'] = df_123['sentence_2'].apply(make_blank)
    else:
        pass

    return pd.concat([data, df_123], ignore_index=True).drop('round-label',axis=1)

if __name__=="__main__":
    test = "이거 잘 되는거 맞아?"
    split_test = split_syllables(test)
    join_test = join_jamos(split_test)
    print(split_test,type(split_test))
    print(join_test,type(join_test))