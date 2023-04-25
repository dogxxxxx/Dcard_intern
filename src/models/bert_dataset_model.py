from transformers import BertTokenizerFast, AutoModel
from torch import nn
import torch

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df.tolist()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text and convert to input IDs and attention masks
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df['title'].tolist()
        self.labels = df['labels'].tolist()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text and convert to input IDs and attention masks
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Convert label to tensor
        label_tensor = torch.tensor(label)
        
        return input_ids, attention_mask, label_tensor

# Define the model
class BertClassifier(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('ckiplab/bert-base-chinese')
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        logits = self.sigmoid(logits)
        return logits

class BertRegressor(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(BertRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained('ckiplab/bert-base-chinese')
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits
