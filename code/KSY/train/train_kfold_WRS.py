import argparse
import random

import pandas as pd
import numpy as np

import wandb

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from sklearn.model_selection import KFold

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, masks, targets=[]):
        self.inputs = inputs
        self.masks = masks
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.masks[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.masks[idx]), torch.tensor(self.targets[idx])


    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, num_split, k):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.sampler = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=256)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.num_split = num_split
        self.k = k

    def tokenizing(self, dataframe):
        data, mask = [], []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
            mask.append(outputs['attention_mask'])
        return data, mask

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs, masks = self.tokenizing(data)

        return inputs, masks, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path, index_col=0).dropna().reset_index()
            val_data = pd.read_csv(self.dev_path)
            all_data = pd.concat([train_data, val_data], ignore_index=True)

            # KFold
            kf = KFold(n_splits=self.num_split, shuffle=True, random_state=42)

            all_split = [k for k in kf.split(all_data)]
            train_idx, valid_idx = all_split[self.k]

            df_train = all_data.iloc[train_idx]
            df_valid = all_data.iloc[valid_idx]

            # 학습데이터 준비
            train_inputs, train_masks, train_targets = self.preprocessing(df_train)

            # 검증데이터 준비
            val_inputs, valid_masks, val_targets = self.preprocessing(df_valid)
            
            
            # 불균형 맞춰주기 - Sampler
            df_train['round-label'] = df_train['label'].apply(lambda x: int(np.round(x)))
            class_counts = df_train['round-label'].value_counts().to_list()
            num_samples = sum(class_counts)
            labels = df_train['round-label'].to_list()

            # 클래스별 가중치 부여 (총 개수/라벨별 개수)
            class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]

            # 해당 데이터 라벨에 대한 가중치
            weights = [class_weights[labels[i]] for i in range(int(num_samples))]
            self.sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_masks, train_targets)
            self.val_dataset = Dataset(val_inputs, valid_masks, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)

            # # 라벨 로그로 변환
            # test_data[self.target_columns[0]] = np.log(test_data[self.target_columns[0]])
            # test_max = test_data[self.target_columns[0]].max()
            # test_data[self.target_columns[0]] = test_data[self.target_columns[0]].apply(lambda x: test_max - x)

            test_inputs, test_masks, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_masks, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_masks, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, predict_masks, [])

    def train_dataloader(self):
        # return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle, num_workers=8)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=8)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()
        #self.loss_func = torch.nn.MSELoss()

    def forward(self, x, masks):
        x = self.plm(input_ids=x, attention_mask=masks)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, masks, y = batch
        logits = self(x,masks)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, masks, y = batch
        logits = self(x, masks)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, masks, y = batch
        logits = self(x, masks)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x, masks = batch
        logits = self(x, masks)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.9)
            # T_max 가 저점을 찍을 지점
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        sch_config = {
            "scheduler": scheduler,
            "interval": "step", # step 단위로 lr 조정하는 것이 좋음
        }
        return [optimizer], [sch_config]

if __name__ == '__main__':

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    def seed_all(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print("SEED VALUE SET:", seed)

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='/data/ephemeral/home/level1/data/aug_train_row_ver2.csv')
    parser.add_argument('--dev_path', default='/data/ephemeral/home/level1/data/dev.csv')
    parser.add_argument('--test_path', default='/data/ephemeral/home/level1/data/dev.csv')
    parser.add_argument('--predict_path', default='/data/ephemeral/home/level1/data/test.csv')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--run_name', type=str)
    # parser.add_argument('--loss_type', default='RMSE', type=str)
    parser.add_argument('--num_split', default=5, type=int)
    args = parser.parse_args(args=[])

    # main에서 불러와서 설정. train.py와 inference.py 모두 설정.
    seed_all(args.seed)
    # dataloader와 model을 생성합니다.
    # dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
    #                         args.test_path, args.predict_path)
    # model = Model(args.model_name, args.learning_rate)

    # wandb.init()
    wandb.login(key=WandB_key)

    logger = WandbLogger(project="level1", group = "kfold+WRS", entity='moth2aflame', name='SNU/ELECTRA')
    logger.experiment.config.update(args)
                                    
    # early_stop_cb = EarlyStopping(monitor="val_loss", patience=10)
    checkpoint_cb = ModelCheckpoint('./checkpoints')
    lr_cb = LearningRateMonitor(logging_interval='step')

    # KFold
    # dataloader와 model을 생성합니다.
    model = Model(args.model_name, args.learning_rate)
    
    for k in range(args.num_split):
        model = Model(args.model_name, args.learning_rate)
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path, args.num_split, k=k)
        
        patience = 3
        filename = 'best_checkpoint'

        early_stop_custom_callback = EarlyStopping(
                "val_pearson", patience=patience, verbose=True, mode="max"
        )

        checkpoint_callback = ModelCheckpoint(
                monitor="val_pearson",
                save_top_k=1,
                dirpath="./callback",
                filename=filename,
                save_weights_only=False,
                verbose=True,
                mode="max",
        )

        learning_rate_monitor_callback = LearningRateMonitor(logging_interval='step')

        # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=args.max_epoch,
            log_every_n_steps=1,
            # val_check_interval=100, # 이 값을 설정하면 1 epoch 안에서 설정한 값 마다 validation 진행
            precision=16, # 더 빠른 학습을 위해, 만약 학습 진행이 안되면 이 값을 뺴보길
            gradient_clip_val=1.0, # 안정적인 학습을 위헤
            callbacks=[early_stop_custom_callback, checkpoint_callback, learning_rate_monitor_callback],
            logger=logger,
        )

        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        torch.save(model, f'/data/ephemeral/home/level1/ksy/outputs/model_kfold_{k}.pt')
    

    # Train part
    # trainer.fit(model=model, datamodule=dataloader)
    # trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    # torch.save(model, '/data/ephemeral/home/level1/AYJ/output/model_kfold.pt')