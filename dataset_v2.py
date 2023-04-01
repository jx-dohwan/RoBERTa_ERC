import json
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, AutoTokenizer
import torch

class data_loader(Dataset):
    def __init__(self,data_path):
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

        with open(train_path) as f:
          data = json.load(f)

        self.session_dataset  = []

        for i in range(len(data)):
          if 10 <= int(data[i]['profile']['emotion']['type'][-2:]) <= 19:
            data[i]['profile']['emotion']['type'] = '분노'
          elif 20 <= int(data[i]['profile']['emotion']['type'][-2:]) <= 29:
            data[i]['profile']['emotion']['type'] = '슬픔'
          elif 30 <= int(data[i]['profile']['emotion']['type'][-2:]) <= 39:
            data[i]['profile']['emotion']['type'] = '불안'
          elif 40 <= int(data[i]['profile']['emotion']['type'][-2:]) <= 49:
            data[i]['profile']['emotion']['type'] = '상처'
          elif 50 <= int(data[i]['profile']['emotion']['type'][-2:]) <= 59:
            data[i]['profile']['emotion']['type'] = '당황'
          elif 60 <= int(data[i]['profile']['emotion']['type'][-2:]) <= 69:
            data[i]['profile']['emotion']['type'] = '기쁨'

        for i in range(len(data)):
          data_values = list(data[i]['talk']['content'].values())
          values = [val for val in data_values if val != '']
          values.append(data[i]['profile']['emotion']['type'])
          self.session_dataset.append(values)

        self.special_words = ["#hh#", "#hs#"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.special_words}) # 나중에 모델부분에서 스폐셜 토큰 갯수를 더 추가해서 resize를 해야함함

        self.emoList = ['분노', '슬픔', '불안', '상처', '당황', '기쁨']

    def __len__(self):
        return len(self.session_dataset)

    def __getitem__(self,idx):
        return self.session_dataset[idx]

    def padding(self, batch_input_token):
        batch_token_ids, batch_attention_masks = batch_input_token['input_ids'], batch_input_token['attention_mask']
        trunc_batch_token_ids, trunc_batch_attention_masks = [], []
        for batch_token_id, batch_attention_mask in zip(batch_token_ids, batch_attention_masks):
            if len(batch_token_id) > self.tokenizer.model_max_length: 
                # 맨 앞에 감정을 담는 인자가 있기 때문에 앞에서 부터 자른다. 
                trunc_batch_token_id = [batch_token_id[0]] + batch_token_id[1:][:self.tokenizer.model_max_length-1]
                trunc_batch_attention_mask = [batch_attention_mask[0]] + batch_attention_mask[1:][:self.tokenizer.model_max_length-1]
                trunc_batch_token_ids.append(trunc_batch_token_id)
                trunc_batch_attention_masks.append(trunc_batch_attention_mask)
            else: # 512보다 짧은 경우
                trunc_batch_token_ids.append(batch_token_id)
                trunc_batch_attention_masks.append(batch_attention_mask)
        """padding token"""
        max_length = max([len(x) for x in trunc_batch_token_ids])
        padding_tokens, padding_attention_masks = [], []
        for batch_token_id, batch_attention_mask in zip(trunc_batch_token_ids, trunc_batch_attention_masks):
            padding_tokens.append(batch_token_id + [self.tokenizer.pad_token_id for _ in range(max_length-len(batch_token_id))])
            padding_attention_masks.append(batch_attention_mask + [0 for _ in range(max_length-len(batch_token_id))])

        return torch.tensor(padding_tokens), torch.tensor(padding_attention_masks)

    def collate_fn(self, sessions):
        batch_input, batch_labels = [], [] # com입력, 학습할 레이블
        batch_PM_input = [] # pm 입력
        for session in sessions: 
            input_str = self.tokenizer.cls_token 
            PM_input = []

            for i in range(len(session[:-1])):
                if i % 2 == 0:
                      input_str += " " + self.special_words[0] + session[i] + self.tokenizer.sep_token
                else:
                     input_str +=  " " + self.special_words[1] + session[i] + self.tokenizer.sep_token
                
                if i != 0:
                    PM_input.append(self.tokenizer.encode(session[i], add_special_tokens=True, return_tensors='pt'))

            batch_input.append(input_str)
            batch_labels.append(self.emoList.index(session[-1]))
            batch_PM_input.append(PM_input)
        batch_input_token = self.tokenizer(batch_input, add_special_tokens=False)
        batch_padding_token, batch_padding_attention_mask = self.padding(batch_input_token)

        return batch_padding_token, batch_padding_attention_mask, batch_PM_input, torch.tensor(batch_labels)         