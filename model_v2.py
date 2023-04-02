from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class ERC_model(nn.Module):
    def __init__(self, clsNum):
        super(ERC_model, self).__init__()
        self.com_model = AutoModel.from_pretrained('klue/roberta-base')
        self.pm_model =  AutoModel.from_pretrained('klue/roberta-base')
        self.tokenizer =  AutoTokenizer.from_pretrained('klue/roberta-base')

        special_words = ["#hh#", "#hs#"] 
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_words}) 
        self.com_model.resize_token_embeddings(len(self.tokenizer))
        self.pm_model.resize_token_embeddings(len(self.tokenizer))

        """GRU"""
        self.hiddenDim = self.com_model.config.hidden_size
        zero = torch.empty(2, 1, self.hiddenDim)
        self.h0 = torch.zeros_like(zero).cuda() # 레이어들은 gpu로 작동하나 이런형식으로 값을 지정하는 것은 cuda()를 따로 지정해줘야한다.
        self.speakerGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3)

        """score matrix"""
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_padding_token, batch_padding_attention_mask, batch_PM_input):
        """for CoM"""
        batch_com_out = self.com_model(input_ids=batch_padding_token, attention_mask=batch_padding_attention_mask)['last_hidden_state']
        batch_com_first = batch_com_out[:,0,:]

        """ GRU 통과 --> PM 결과 """
        batch_pm_gru_first = []
        for PM_inputs in batch_PM_input:
            if PM_inputs:
                pm_outs = []
                for PM_input in PM_inputs:
                    pm_out = self.pm_model(PM_input)['last_hidden_state'][:,0,:] # CLS의 출력/attention에 해당하는 것을 명시하지 않고 토큰들만 넣어 pm 출력 뽑아냄 그중 CLS
                    pm_outs.append(pm_out)
                pm_outs = torch.cat(pm_outs, 0).unsqueeze(1) # (speaker_num, batch=1, hidden_dim)로 만듬 토치텐서
                pm_gru_outs, _ = self.speakerGRU(pm_outs, h0) # (speaker_num, batch=1, hidden_dim)로 만듬 토치텐서서 pm_outs의 hs 계산하기전 h0로 초기화하는 것/ 그리고 model의 hs와 현재 발화자의 값을 가져와 hs를 업데이트 하는 것이다.
                pm_gru_first = pm_gru_outs[0,:,:] # (1, hidden_dim) 첫번째것이이 중요하니 해당하는 것을 가져와서 사용함
                batch_pm_gru_first.append(pm_gru_first)
            else:
                batch_pm_gru_first.append(torch.zeros(1, hiddenDim).cuda()) #pm입력이 없는 경우 torch zero를 넣어준다.
        batch_pm_gru_first = torch.cat(batch_pm_gru_first, 0)

        """score matrix"""
        first_output = self.W(batch_com_first + batch_pm_gru_first)

        return first_output