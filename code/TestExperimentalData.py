import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

###Covid19 data
Covid19_datapath=r'data\ExperimentalData\Covid19.csv'
Covid19_raw=pd.read_csv(Covid19_datapath,sep='\t')

###PDCD1_8 data
PDCD1_8_datapath=r'data\ExperimentalData\PDCD1_8.csv'
PDCD1_8_raw=pd.read_csv(PDCD1_8_datapath,sep='\t')
print(PDCD1_8_raw)

###Get 6-mer siRNA for test dataset
def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers
    
Covid19_text=Covid19_raw.values[:,1]
Covid19_test_6mer=[seq2kmer(i,6) for i in Covid19_text]
print(Covid19_test_6mer[0])
Covid19_test_label=list(Covid19_raw.values[:,4])
Covid19_test=pd.DataFrame({'scores':Covid19_test_label ,'text':Covid19_test_6mer})
print(Covid19_test)

PDCD1_8_text=PDCD1_8_raw.values[:,1]
PDCD1_8_test_6mer=[seq2kmer(i,6) for i in PDCD1_8_text]
print(PDCD1_8_test_6mer[0])
PDCD1_8_test_label=list(PDCD1_8_raw.values[:,4])
PDCD1_8_test=pd.DataFrame({'scores':PDCD1_8_test_label ,'text':PDCD1_8_test_6mer})
print(PDCD1_8_test)

###Construct BERT-based model

import torch
from torch import nn
import numpy as np
from transformers import BertModel, BertConfig,BertTokenizer

tokenizer = BertTokenizer.from_pretrained(r'.\DNABERTbased\6mer')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels =[label for label in df['scores']]
        self.texts = [tokenizer(text, padding='max_length', max_length = 16, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_y=torch.tensor(batch_y, dtype=torch.float32)
        
        return batch_texts, batch_y

class siRNABertRegressor(nn.Module):

    def __init__(self, dropout=0.1):

        super(siRNABertRegressor, self).__init__()
      
        config=BertConfig.from_pretrained(r'.\DNABERTbased\6mer', output_attentions=True)
        self.bert = BertModel.from_pretrained(r'.\DNABERTbased\6mer',config=config)      
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(64, 1)
        self.sigmoid4 = nn.Sigmoid()
    def forward(self, input_id, mask):

        output = self.bert(input_ids= input_id, attention_mask=mask)
        attention=output[-1]
        label_output=output[0][:,0,:]
        dropout_output = self.dropout(label_output)
        linear_output1 = self.linear1(dropout_output)
        reluoutput1=self.relu1(linear_output1)
        linear_output2=self.linear2(reluoutput1)
        reluoutput2=self.relu2(linear_output2)
        linear_output3=self.linear3(reluoutput2)
        reluoutput3=self.relu3(linear_output3)
        linear_output4=self.linear4(reluoutput3)
        final_layer = self.sigmoid4(linear_output4)

        return final_layer,attention,reluoutput1,reluoutput2,reluoutput3
    
model = siRNABertRegressor()

###Load and test
from sklearn.metrics import roc_curve, auc
classify=0.7
def evaluate(ytrue,pred): 
    out=np.vstack((pred,ytrue)).T
    dat=pd.DataFrame(out)
    pcc1=dat.corr().iloc[0,1]
    pcc2=dat.corr('spearman').iloc[0,1]
    return pcc1,pcc2
seed = 17
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
test_dataloader=torch.utils.data.DataLoader(Dataset(PDCD1_8_test),batch_size=1000)
model=torch.load(r'.\model\version8\model_(4Layer_5e-5_dropout0.1_reg1e-2).pt')
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
        model = model.cuda()
                 
l_test=[]
p_test=[]
with torch.no_grad():
    for test_input, test_label in test_dataloader:
                 
        test_label = test_label.to(device)
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)

        test_output,attention,last3,last2,last1 = model(input_id, mask)

        y_test=[round(i,3) for i in test_label.tolist()]
        predictions=[i[0] for i in test_output.tolist()]
        correlation_test=evaluate(y_test,predictions)
                 
print(f'Test Correlation: {correlation_test} ')