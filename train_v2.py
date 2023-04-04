import torch
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoConfig
from tqdm import tqdm
import argparse
import pdb
import os
import sys
import torch.nn as nn
sys.path.append('/content/drive/MyDrive/인공지능/대화속감정인식')
from dataset_v2 import data_loader
from model_v2 import ERC_model
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import logging
import json
import pandas as pd
# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('erc.log')
logger.addHandler(file_handler)

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--train_fn', required=True)
    p.add_argument('--dev_fn', required=True)
    p.add_argument('--test_fn', required=True)
    p.add_argument('--save_fn', required=True)
    
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--training_steps', type=int, required=False)
    p.add_argument('--warmup_steps', type=int, required=False)
    p.add_argument('--grad_norm', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=1) # batch_size는 1로해야지 돌아감 
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-6)    

    config = p.parse_args()

    return config

def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

# https://tutorials.pytorch.kr/beginner/saving_loading_models.html
def SaveModel(model, path, tokenizer, config, optimizer, scheduler, epoch, loss, test_fbeta):
    path = os.path.join(path, "checkpoint-{}".format(epoch+1))
    if not os.path.exists(path):
        os.makedirs(path)
    
    tokenizer.save_pretrained(path)
    config.save_pretrained(path)

    # torch.save(model.state_dict(), os.path.join(path, "ERC_model.bin"))
    # logger.info("Saving model checkpoint to %s", path)

    # torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    # torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
    
    # config_dict = config.to_dict()
    # with open(path+"/config.json", "w", encoding='utf-8') as f:
    #     json.dump(config_dict, f)
    # logger.info("Saving optimizer and scheduler states to %s", path)
    df = pd.DataFrame({"Test W-avg F1" : [test_fbeta]})
    df.to_csv(os.path.join(path, "F1_W_avg.csv"), mode='w', index = False)
    torch.save({
        'epoch' : epoch,
        'ERC_model' : model.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'config' : config,
        'optimizer' : optimizer.state_dict(),
        'loss' : loss
    },  os.path.join(path, 'ERC_model.bin'))


def CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_padding_token, batch_padding_attention_mask, batch_PM_input, batch_label = data
            batch_padding_token = batch_padding_token.cuda()
            batch_padding_attention_mask = batch_padding_attention_mask.cuda()
            batch_PM_input = [[x2.cuda() for x2 in x1] for x1 in batch_PM_input]
            batch_label = batch_label.cuda()        

            """Prediction"""
            pred_logits, none_data = model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_label.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)
    return acc, pred_list, label_list    

def config_json_file(config, vocab_size):
  
    config_json = AutoConfig.from_pretrained('klue/roberta-base')
    config_json.layer_norm_eps = config.lr
    config_json.vocab_size = vocab_size
    config_json.grad_norm = config.grad_norm

    return config_json

def main(config):

    tokenizer =  AutoTokenizer.from_pretrained('klue/roberta-base')
    
    train_dataset = data_loader(config.train_fn)
    dev_dataset = data_loader(config.dev_fn)
    test_dataset = data_loader(config.test_fn)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=test_dataset.collate_fn)

    clsNum = len(train_dataset.emoList)
    erc_model = ERC_model(clsNum).cuda()
  
    if config.training_steps is not None and config.warmup_steps is not None:
        num_training_steps = config.training_steps
        num_warmup_steps = config.warmup_steps
    elif config.training_steps is not None and config.warmup_steps is None:
        num_training_steps = config.training_steps
        num_warmup_steps = len(train_dataset)
    elif config.training_steps is None and config.warmup_steps is not None:
        num_training_steps = len(train_dataset) * config.epochs
        num_warmup_steps = config.warmup_steps
    else:
        num_training_steps = len(train_dataset) * config.epochs
        num_warmup_steps = len(train_dataset)
    
    optimizer = torch.optim.AdamW(erc_model.parameters(), lr=config.lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    logger.info("##########학습 시작##########")
    best_dev_fscore = 0
    for epoch in tqdm(range(config.epochs)):
        erc_model.train()
        for i_batch, data in enumerate(tqdm(train_dataloader)):
            batch_padding_token, batch_padding_attention_mask, batch_PM_input, batch_label = data
            batch_padding_token = batch_padding_token.cuda()
            batch_padding_attention_mask = batch_padding_attention_mask.cuda()
            batch_PM_input = [[x2.cuda() for x2 in x1] for x1 in batch_PM_input]
            batch_label = batch_label.cuda()        

            """Prediction"""
            pred_logits, vocab_size = erc_model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)

            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, batch_label)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(erc_model.parameters(), config.grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        """dev & test evaluation"""
        erc_model.eval()

        dev_acc, dev_pred_list, dev_label_list = CalACC(erc_model, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        
        logger.info("Dev W-avg F1: {}".format(dev_fbeta))
        
        test_acc, test_pred_list, test_label_list = CalACC(erc_model, test_dataloader)
        """Best Score & Model Save"""
        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta

            test_acc, test_pred_list, test_label_list = CalACC(erc_model, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                

            config_json = config_json_file(config, vocab_size)

            SaveModel(erc_model, config.save_fn, tokenizer, config_json, optimizer, scheduler, epoch, loss_val, test_fbeta)
            logger.info("Epoch:{}, Test W-avg F1: {}".format(epoch, test_fbeta))


if __name__ == '__main__':
    config = define_argparser()
    main(config)
