import torch
import sys
sys.path.append('/content/drive/MyDrive/인공지능/대화속감정인식')
from dataset_v2 import data_loader
from torch.utils.data import DataLoader
from model_v2 import ERC_model
from tqdm import tqdm

def ErrorSamples(data_path, model_path):

    test_dataset = data_loader(data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)

    clsNum = len(test_dataset.emoList)
    erc_model = ERC_model(clsNum).cuda()
    checkpoint = torch.load(model_path)
    erc_model.load_state_dict(checkpoint['ERC_model'])

    erc_model.eval()
    correct = 0
    label_list = []
    pred_list = []    
    
    error_samples = []
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(test_dataloader)):
            """Prediction"""
            batch_padding_token, batch_padding_attention_mask, batch_PM_input, batch_label = data
            batch_padding_token = batch_padding_token.cuda()
            batch_padding_attention_mask = batch_padding_attention_mask.cuda()
            batch_PM_input = [[x2.cuda() for x2 in x1] for x1 in batch_PM_input]
            batch_label = batch_label.cuda()        

            """Prediction"""
            pred_logits,_ = erc_model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_label.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)            
            if pred_label != true_label:
                error_samples.append([batch_padding_token, true_label, pred_label])
            if pred_label == true_label:
                correct += 1
        acc = correct/len(test_dataloader)                
    return error_samples, acc, pred_list, label_list     