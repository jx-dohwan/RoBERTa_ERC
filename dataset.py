import csv
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import torch

def split(session):
    final_data = []
    split_session =[]
    for line in session:
        split_session.append(line)
        final_data.append(split_session[:])
    return final_data

class data_loader(Dataset):
    def __init__(self, data_path):
        f = open(data_path, 'r')
        rdr = csv.reader(f)
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        emoSet = set()        

        """ 세션 데이터 저장할 것"""
        self.session_dataset = []
        session = []
        speaker_set = []

        """ 실제 데이터 저장 방식 """
        pre_sess = 'start'
        for i, line in enumerate(rdr):
            if i == 0:
                """ 저장할 데이터들 index 확인 """
                header  = line
                utt_idx = header.index('Utterance')
                speaker_idx = header.index('Speaker')
                emo_idx = header.index('Emotion')
                sess_idx = header.index('Dialogue_ID')
            else:
                utt = line[utt_idx]
                speaker = line[speaker_idx]
                """ 유니크한 스피커로 바꾸기 """
                if speaker in speaker_set:# 스키퍼셋안에 있는 스피커(즉 이미 있는 스피커라면)
                    uniq_speaker = speaker_set.index(speaker)# 어떤 사람이 말했는지 그 사람의 index를 uniq로 둔다.
                else:# 처음 등장한 인물이라면
                    speaker_set.append(speaker)# 스피커셋에 스피커를 넣어주고
                    uniq_speaker = speaker_set.index(speaker)#유니크 스피커가 몇 번째 인덱스에 해당하는지 바꿔줌
                emotion = line[emo_idx]
                sess = line[sess_idx]
                
                if pre_sess == 'start' or sess == pre_sess:# sess이 pre_sess이 같거나 start일때 저장
                    session.append([uniq_speaker, utt, emotion])
                else:#다를경우 초기화하고 다시저장?
                    """ 세션 데이터 저장 """
                    # session_dataset.append(session)
                    self.session_dataset += split(session)# 서브세셔으로 분리하여 데이터로 바꿈
                    session = [[uniq_speaker, utt, emotion]]
                    speaker_set = []
                    emoSet.add(emotion)
                pre_sess = sess   
        """ 마지막 세션 저장 """
        # session_dataset.append(session)
        self.session_dataset += split(session)

        # self.emoList = sorted(emoSet) # 항상 같은 레이블 순서를 유지하기 위해
        self.emoList = ['anger', 'disgust', 'fear',' joy', 'neutral', 'sadness', 'surprise']
        f.close()
    
    def __len__(self): # Dataset 기본적인 구성
        return len(self.session_dataset) ## 리스트의 길이이

    def __getitem__(self, idx): #기본적인 구성
        return self.session_dataset[idx] ## 리스트중 하나를 불러와서 사용함

    def padding(self, batch_input_token):# batch가 1이면 필요없다./ 길이가 다른 입력을 처리할 필요가없기 때문에
        """추가"""
        """ 512 토큰 길이 넘으면 잘라내기"""
        batch_token_ids, batch_attention_masks = batch_input_token['input_ids'], batch_input_token['attention_mask'] # 
        trunc_batch_token_ids, trunc_batch_attention_masks = [], []
        for batch_token_id, batch_attention_mask in zip(batch_token_ids, batch_attention_masks):
            if len(batch_token_id) > self.tokenizer.model_max_length: # 512보다 길이가 긴경우
                trunc_batch_token_id = [batch_token_id[0]] + batch_token_id[1:][-self.tokenizer.model_max_length+1:] 
                # cls는 필수, 1부터 길이만큼인데 뒤부분부터 자름 앞부분부터 자르면 마지막 발화에 감정 예측 토큰이 사라지면 학습이 제대로 되지 않는다. 
                trunc_batch_attention_mask = [batch_attention_mask[0]] + batch_attention_mask[1:][-self.tokenizer.model_max_length+1:]
                # attention mask도 위와 동일하게
                trunc_batch_token_ids.append(trunc_batch_token_id)
                trunc_batch_attention_masks.append(trunc_batch_attention_mask)
            else: # 512보다 길이가 짧은 경우 -> 그냥 들어간다.
                trunc_batch_token_ids.append(batch_token_id)
                trunc_batch_attention_masks.append(batch_attention_mask)
        """ padding token으로 패딩하기"""
        #
        max_length = max([len(x) for x in trunc_batch_token_ids])
        padding_tokens, padding_attention_masks = [], []
        for batch_token_id, batch_attention_mask in zip(trunc_batch_token_ids, trunc_batch_attention_masks):
            padding_tokens.append(batch_token_id + [self.tokenizer.pad_token_id for _ in range(max_length-len(batch_token_id))])
            # 512-max_length해서 남는 부분을 padding 토큰 입력
            padding_attention_masks.append(batch_attention_mask + [0 for _ in range(max_length-len(batch_token_id))])
            # 0을 넣어준다.
        return torch.tensor(padding_tokens), torch.tensor(padding_attention_masks) # 리스트가 텐서포 변함

    def collate_fn(self, sessions): # 배치를 위한 구성, sessions는 session_dataset모두 들어옴
        '''
            input:
                data : [(session1),(session2),...]
            return:
                batch_input_tokens_pad: (B, L) apdded
                batch_labels: (B)
        '''
        # 컨텍스트 길이 조정해도 된다.
        # 발화 1이런식으로 앞에 제거할 수 있다. 
        # 앞을 제구하는 이유는 뒤에 내용이 더 중요하기 때문이다.
        """추가"""
        batch_input, batch_labels = [], [] # com 입력, 학습할 레이블
        batch_PM_input = [] # pm 입력
        for session in sessions: # 하나의 세션을 linebyline으로 쪼개서 input_token으로 만든다.
            input_str = self.tokenizer.cls_token # 입력의 맨 앞에 들어감
            """For PM"""
            current_speaker, curent_utt, current_emotion = session[-1] # 마지막을 의미함/마지막 발화자를 알기 위함
            PM_input = [] 
            for i, line in enumerate(session): # 세션에서 하나씩 부르는데 
                speaker, utt, emotion = line
                input_str += " " + utt + self.tokenizer.sep_token # 중첩해서 더해줌 구분은 sep로 
                if i < len(session)-1 and current_speaker == speaker: # 마지막은 무조건 스피커와 커런트가 같으니까 아닌 경우에 대해 입력해줌/스피커와 커런트 스피커와 같으면 입력
                    PM_input.append(self.tokenizer.encode(utt, add_special_tokens=True, return_tensors='pt')) 
                    # True : concat해서 저장하는 것이 아니고 하나의 입력을 바로 PM에 태우는 것이기 때문
                    # [cls_token, tokens, sep_token]
            
            """For CoM"""
            batch_input.append(input_str)
            batch_labels.append(self.emoList.index(emotion))
            batch_PM_input.append(PM_input) # batch안에 batch의 개념
        batch_input_token = self.tokenizer(batch_input, add_special_tokens=False)
        batch_padding_token, batch_padding_attention_mask = self.padding(batch_input_token)

        return batch_padding_token, batch_padding_attention_mask, batch_PM_input, torch.tensor(batch_labels)      