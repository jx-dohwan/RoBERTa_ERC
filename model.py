from transformers import RobertaModel
import torch
import torch.nn as nn
import pdb

class ERC_model(nn.Module):
    def __init__(self, clsNum):
        super(ERC_model, self).__init__()
        self.com_model = RobertaModel.from_pretrained('roberta-base')
        self.pm_model = RobertaModel.from_pretrained('roberta-base')
        
        """ GRU 세팅 """
        self.hiddenDim = self.com_model.config.hidden_size
        zero = torch.empty(2, 1, self.hiddenDim)
        self.h0 = torch.zeros_like(zero).cuda() # (num_layers * num_directions, batch, hidden_size)
        self.speakerGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)
        
        """ score matrix """
        self.W = nn.Linear(self.hiddenDim, clsNum)
    def forward(self, batch_padding_token, batch_padding_attention_mask, batch_PM_input):
        """ for CoM """
        batch_com_out = self.com_model(input_ids=batch_padding_token, attention_mask=batch_padding_attention_mask)['last_hidden_state']
        batch_com_final = batch_com_out[:,0,:]
        
        """ GRU 통과 --> PM 결과 """
        batch_pm_gru_final = []
        for PM_inputs in batch_PM_input:
            if PM_inputs:
                pm_outs = []
                for PM_input in PM_inputs:
                    pm_out = self.pm_model(PM_input)['last_hidden_state'][:,0,:]
                    pm_outs.append(pm_out)
                pm_outs = torch.cat(pm_outs, 0).unsqueeze(1) # (speaker_num, batch=1, hidden_dim)
                pm_gru_outs, _ = self.speakerGRU(pm_outs, self.h0) # (speaker_num, batch=1, hidden_dim)
                pm_gru_final = pm_gru_outs[-1,:,:] # (1, hidden_dim)
                batch_pm_gru_final.append(pm_gru_final)
            else:
                batch_pm_gru_final.append(torch.zeros(1, self.hiddenDim).cuda())
        batch_pm_gru_final = torch.cat(batch_pm_gru_final, 0)        
        
        """ score matrix """
        #pdb.set_trace()
        final_output = self.W(batch_com_final + batch_pm_gru_final) # (B, C)
        
        return final_output