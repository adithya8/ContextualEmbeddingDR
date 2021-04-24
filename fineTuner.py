import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine

import os
import sys
import json
import argparse
from pprint import pprint
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
from sklearn.metrics import f1_score
#from sklearn.preprocessing import OneHotEncoder

#from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig

TASK_PARAM_DICT = {
    "age": {
        "outcome": "age",
        "message_table": "T_20",
        "outcome_table": "20_outcomes",
        "correl_field": "user_id",
        "regression": True,
        "num_classes": None,
    },
    "gen": {
        "outcome": "gender",
        "message_table": "T_20",
        "outcome_table": "20_outcomes",
        "correl_field": "user_id",
        "regression": False,
        "num_classes": 2,
    },
    "gen2": {
        "outcome": "cntrl_gender",
        "message_table": "T_18",
        "outcome_table": "18_outcomes",
        "correl_field": "clp18_id",
        "regression": False,
        "num_classes": 2,
    },
    "ext": {
        "outcome": "ext",
        "message_table": "T_20",
        "outcome_table": "20_outcomes",
        "correl_field": "user_id",
        "regression": True,
        "num_classes": None,
    },
    "ope": {
        "outcome": "ope",
        "message_table": "T_20",
        "outcome_table": "20_outcomes",
        "correl_field": "user_id",
        "regression": True,
        "num_classes": None,
    },
    "bsag": {
        "outcome": "a11_bsag_total",
        "message_table": "T_18",
        "outcome_table": "18_outcomes",
        "correl_field": "clp18_id",
        "regression": True,
        "num_classes": None,
    },
    "sui": {
        "outcome": "label",
        "message_table": "T_19",
        "outcome_table": "19_outcomes",
        "correl_field": "user_id",
        "regression": False,
        "num_classes": 4,
    },
}

class LineByLineTextDataset(Dataset):
    def __init__(self, query, tokenizer, num_classes:int=1):
        def get_lines(num_classes):
            lines = conn.execute(query).fetchall()
            user_id = list(map(lambda x: x[0], lines))
            target = list(map(lambda x: x[2], lines))
            lines = list(map(lambda x: x[1], lines))
            print (f"Retrieved: {len(user_id)}")
            new_users, new_lines, new_target = [], [], []
            with tqdm(total = len(lines)) as trainer:
                trainer.set_description('Loading Data....')
                for line in range(len(lines)):
                    if(lines[line]):
                        if (len(lines[line]) > 0 and not lines[line].isspace()):
                            new_users.append(user_id[line])
                            new_lines.append(lines[line])
                            new_target.append(target[line])
                    trainer.update(1)
            
            #if num_classes>=3: 
                #new_target = OneHotEncoder().fit(np.arange(num_classes).reshape(-1,1)).transform(np.array(new_target).reshape(-1,1))
            return new_users, new_lines, new_target
    
        mydb = URL(drivername='mysql', host='localhost', database="dimRed_contextualEmb", query={'read_default_file':'~/.my.cnf'})
        engine = create_engine(mydb)
        conn = engine.connect()
        
        
        print (f"Query: {query}")
        self.users, lines, self.target = get_lines(num_classes)
        tokenized = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=512)
        self.input_tokens = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, i):
        return (self.users[i], \
                torch.tensor(self.input_tokens[i]), \
                torch.tensor(self.attention_mask[i]), \
                torch.tensor(float(self.target[i])))


class LMFineTuner(pl.LightningModule):

    def __init__(self, hparams):
        super(LMFineTuner, self).__init__()
        
        self.hparams = hparams

        print ('----------------------')
        pprint (self.hparams)
        print ('----------------------')        

        self.target = "age" if self.hparams.regression == True else "gender"
        
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.config = AutoConfig.from_pretrained("roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.contextualModel = AutoModel.from_config(self.config)
        #self.contextualModel = BertModel.from_pretrained('bert-base-uncased')
        self.ftLayer = nn.Linear(768, 1) if self.hparams.regression == True else nn.Linear(768, self.hparams.num_classes)
        self.activation = torch.nn.ReLU()
        self.lossFunc = nn.MSELoss(reduction='mean') if self.hparams.regression == True else nn.CrossEntropyLoss()
    
    
    def forward(self, input_ids, attention_mask, labels):

        embeddings = self.contextualModel(input_ids, attention_mask)[0]        
        pooler = (embeddings*attention_mask.view(embeddings.shape[0], embeddings.shape[1], -1).expand(-1, -1, 768))
        pooler = torch.sum(pooler, dim=1)/ torch.sum(attention_mask, dim=1).view(-1, 1).expand(-1, 768)
        op = self.ftLayer(self.activation(pooler)) if self.hparams.regression==True else (self.ftLayer(pooler))
        if self.hparams.regression:
            loss = self.lossFunc(op.view(-1,1).float(), labels.view(-1,1).float())  
        else: 
            loss = self.lossFunc(op, labels.view(-1,).long())
        
        return op, loss
    
    # From mmatero: Test this later
    def configure_optimizers(self):
        """
            Defines otpimizers/schedulers
        """
        model = self
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()        
    
    def training_step(self, batch, batch_idx):
        users, padded, attn, target = batch
        op, loss = self(padded, attn, target)
        loss = loss.unsqueeze(0)
    
        # tensorboard not configured yet
        tb_logs = {'loss': loss}
        #if loss.item() > 1.0:
        #   print (self.trainer.global_step)

        return {'loss': loss, "log": tb_logs}
    
    def validation_step(self, batch, batch_idx):
        users, padded, attn, target = batch
        
        op, loss = self(padded, attn, target)
        scores_dict = {}
        #scores_dict['metric'] = self.metrics(op, target)
        scores_dict['val_loss'] = loss

        scores = [score.unsqueeze(0) for score in scores_dict.values()]
        scores_dict = {key: value for key,value in zip(scores_dict.keys(), scores)}
        scores_dict["users"] = users
        scores_dict["pred"] = op
        scores_dict["target"] = target
        return scores_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs, val="True"):
        result = {}
        val_loss = []
        for batch_op in outputs:
            val_loss.append(batch_op['val_loss'])
            for i in range(len(batch_op['users'])):
                user = batch_op['users'][i]
                if user not in result:
                    result[user] = {}                    
                    result[user]['pred'] = [batch_op['pred'][i][0].item(),]  if self.hparams.regression else [torch.argmax(batch_op['pred'][i]).item(),]  
                    result[user]['target'] = batch_op['target'][i].item()
                else:
                    if self.hparams.regression:
                        result[user]['pred'].append(batch_op['pred'][i][0].item())
                    else:
                        result[user]['pred'].append(torch.argmax(batch_op['pred'][i]).item())
        #print (val_loss)
        
        val_loss = torch.flatten(torch.stack(val_loss))
        val_loss = torch.mean(val_loss)
        pred_median, pred_mean, pred_max, target = [], [], [], []
        for user in result:
            result[user]['user_pred_median'] = np.median(result[user]['pred'])
            result[user]['user_pred_mean'] = np.mean(result[user]['pred'])
            result[user]['user_pred_max'] = np.max(result[user]['pred'])
            pred_median.append(result[user]['user_pred_median'])
            pred_mean.append(result[user]['user_pred_mean'])
            pred_max.append(result[user]['user_pred_max'])
            target.append(result[user]['target'])
            
        if self.hparams.regression:
            metrics = {'progress_bar':{'val_loss': val_loss.detach().cpu().numpy().tolist(), \
                                    'pearson-r (median)': np.corrcoef(pred_median, target)[0,1], \
                                    'pearson-r (mean)': np.corrcoef(pred_mean, target)[0,1], \
                                    #'MSE (median)': np.sum((np.array(pred_median) - np.array(target))**2)/len(pred_median), \
                                    #'MSE (mean)': np.sum((np.array(pred_mean) - np.array(target))**2)/len(pred_mean) \
                                    }} 
        else:
            metrics = {'progress_bar':{ 'val_loss': val_loss.detach().cpu().numpy().tolist(), \
                                        'F1 (median)': f1_score(target, np.around(pred_median), average="macro"), \
                                        'F1 (max)': f1_score(target, pred_max, average="macro"),\
                                        #'Accuracy (median)': np.sum(np.array(pred_median) == np.array(target))/len(pred_median), \
                                        #'Accuracy (mean)': np.sum(np.array(pred_mean) == np.array(target))/len(pred_mean)\
                                    }} 
        dumps = metrics

        with open(f'{self.hparams.output_file}', 'a') as fp:    
            dumps["val"] = val
            json.dump(dumps, fp)
                                        
        return metrics

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, False)
    
    def collate(self, instances):
        return ([i[0] for i in instances], \
                pad_sequence([i[1] for i in instances], batch_first=True, padding_value=self.tokenizer.pad_token_id), \
                pad_sequence([i[2] for i in instances], batch_first=True, padding_value=self.tokenizer.pad_token_id), \
                torch.tensor([i[3] for i in instances]) )

    def train_dataloader(self):
        query = f'''
                SELECT a.{self.hparams.correl_field}, a.message, b.{self.hparams.outcome} 
                FROM {self.hparams.message_table} AS a 
                INNER JOIN 
                (SELECT {self.hparams.correl_field}, {self.hparams.outcome} FROM {self.hparams.outcome_table} 
                WHERE {self.hparams.outcome} IS NOT NULL AND r10pct_test_fold = 0 ORDER BY rand({self.hparams.rand}) 
                LIMIT {self.hparams.num_users}) AS b 
                ON a.{self.hparams.correl_field} = b.{self.hparams.correl_field}
                '''
        train_data = LineByLineTextDataset(query = query, tokenizer=self.tokenizer)
        #train_sampler = RandomSampler(train_data) if self.hparams.distributed_backend != 'ddp' else DistributedSampler(train_data)
        return DataLoader(train_data, batch_size=self.hparams.train_batch_size, shuffle=False, sampler=None, collate_fn=self.collate, num_workers=12)
    
    def test_dataloader(self):
        query = f'''
                SELECT a.{self.hparams.correl_field}, a.message, b.{self.hparams.outcome} 
                FROM {self.hparams.message_table} AS a 
                INNER JOIN 
                (SELECT {self.hparams.correl_field}, {self.hparams.outcome} FROM {self.hparams.outcome_table} 
                WHERE {self.hparams.outcome} IS NOT NULL AND facet_fold IS NOT NULL) AS b 
                ON a.{self.hparams.correl_field} = b.{self.hparams.correl_field};
                '''
        test_data = LineByLineTextDataset(query = query, tokenizer=self.tokenizer, num_classes=self.hparams.num_classes)
        #test_sampler = RandomSampler(test_data) if self.hparams.distributed_backend != 'ddp' else DistributedSampler(test_data)
        return DataLoader(test_data, batch_size=self.hparams.test_batch_size, shuffle=False, sampler=None, collate_fn=self.collate, num_workers=12)
    
    def val_dataloader(self):
        query = f'''
                SELECT a.{self.hparams.correl_field}, a.message, b.{self.hparams.outcome} 
                FROM {self.hparams.message_table} AS a 
                INNER JOIN 
                (SELECT {self.hparams.correl_field}, {self.hparams.outcome} FROM {self.hparams.outcome_table} 
                WHERE {self.hparams.outcome} IS NOT NULL AND r10pct_test_fold = 1) AS b 
                ON a.{self.hparams.correl_field} = b.{self.hparams.correl_field};
                '''
        train_data = LineByLineTextDataset(query = query, tokenizer=self.tokenizer, num_classes=self.hparams.num_classes)
        #train_sampler = RandomSampler(train_data) if self.hparams.distributed_backend != 'ddp' else DistributedSampler(train_data)
        return DataLoader(train_data, batch_size=self.hparams.train_batch_size, shuffle=False, sampler=None, collate_fn=self.collate, num_workers=12)
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()

    #args.add_argument('--regression',  action="store_true", help="Regression or Classification")
    #args.add_argument('--num_classes',  type=int, help="Number of classes for classification")
    args.add_argument('--task',  type=str, help="Task name in the TASK_PARAM_DICT", choices=list(TASK_PARAM_DICT.keys()))
    args.add_argument('--num_users',  type=int, help="Number of users to sample for bootstrapping", default=200)
    args.add_argument('--num_runs',  type=int, help="Number of times to bootstrap sample.", default=1)
    args.add_argument('-e', type=int, default=10, help="Number of Epochs")
    args.add_argument('--lr', type=float, default=3e-5, help="Learning Rate")
    args.add_argument('--adam-epsilon', type=float, default=1e-6, help="Adam Epsilon")
    args.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    args.add_argument('--train-batch-size', type=int, default=3, help="Set the training Batch size")
    args.add_argument('--test-batch-size', type=int, default=3, help="Set the training Batch size")
    args.add_argument('--gpus', type=str, default='0,1,2', help='GPU IDs as CSV')
    args.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'), help="distributed processing protocol")
    args.add_argument('--ckpt-path', type=str, required=True, help='Path to store model ckpt')
    args.add_argument('--output_file', type=str, required=True, help='Path to store results')
    

    args = args.parse_args()
    for i in TASK_PARAM_DICT[args.task]:
        args.__dict__[i] = TASK_PARAM_DICT[args.task][i]

    if (not os.path.exists(args.ckpt_path)):
        print ('Creating CKPT Dir')
        os.mkdir(args.ckpt_path)
    
    '''
    checkpoint_callback = ModelCheckpoint(
        filepath=args.ckpt_path,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    '''
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00005,
        patience=3,
        verbose=False,
        mode='min'
    )

    for trial in range(args.num_runs):
        args.__dict__["rand"] = trial+1

        trainer = pl.Trainer(default_save_path=args.ckpt_path,
                    distributed_backend=args.distributed_backend,
                gpus=len(args.gpus.split(',')), max_epochs=args.e, 
                early_stop_callback=early_stop_callback)
        
        model = LMFineTuner(args)
        trainer.fit(model)
        trainer.test(model)

