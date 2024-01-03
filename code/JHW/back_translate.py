import pandas as pd
from googletrans import Translator
from tqdm import tqdm

class Google_Translator:
    def __init__(self):
        self.translator = Translator()
        self.result = {'src_text': '', 'src_lang': '', 'tgt_text': '', 'tgt_lang': ''}
 
    def translate(self, text, lang='en'):
        translated = self.translator.translate(text, dest=lang)
        self.result['src_text'] = translated.origin
        self.result['src_lang'] = translated.src
        self.result['tgt_text'] = translated.text
        self.result['tgt_lang'] = translated.dest
 
        return self.result
 
    def translate_file(self, file_path, lang='en'):
        with open(file_path, 'r') as f:
            text = f.read()
        return self.translate(text, lang)
    
def back_translate(sentence):
    return translator.translate(translator.translate(sentence, 'en')['tgt_text'], 'ko')['tgt_text']

if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    translator = Google_Translator()
    df_train['b_sentence_1'] = [None] * df_train.shape[0]
    df_train['b_sentence_2'] = [None] * df_train.shape[0]
    for i, row in tqdm(df_train.iterrows()):
        try:
            back_text = back_translate(row['sentence_1'])
            df_train.loc[i,'b_sentence_1'] = back_text
        except:
            print("FAIL in sentence_1 : ", row['id'])
        try:
            back_text = back_translate(row['sentence_2'])
            df_train.loc[i,'b_sentence_2'] = back_text
        except:
            print("FAIL in sentence_2 : ", row['id'])

    df_train.to_csv('../data/train_back.csv')