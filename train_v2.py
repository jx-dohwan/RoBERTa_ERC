import torch
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
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

import logging

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
    p.add_argument('--grad_norm', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=1)
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

def SaveModel(model, path, tokenizer, config, optimizer, scheduler):
    if not os.path.exists(path):
        os.makedirs(path)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(config, model.state_dict(), os.path.join(path, 'model.bin'))    
    logger.info("Saving model checkpoint to %s", path)

    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", path)

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
            pred_logits = erc_model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_label.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)
    return acc, pred_list, label_list

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

    num_training_steps = len(train_dataset)*config.epochs
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
            pred_logits = erc_model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)

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

            SaveModel(erc_model, config.save_path, tokenizer, config, optimizer, scheduler)
            logger.info("Epoch:{}, Test W-avg F1: {}".format(epoch, test_fbeta))


if __name__ == '__main__':
    config = define_argparser()
    main(config)
