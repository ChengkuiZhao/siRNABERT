import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy import mean
import torch
from torch import nn
import numpy as np
from transformers import BertModel, BertConfig,BertTokenizer
from torch.optim import Adam
from tqdm import tqdm

###Huesken  dataset 
datapath=r'data\huesken.csv'
df_raw=pd.read_csv(datapath,sep=',')
print(df_raw.head)
print(df_raw.shape[0])

###Get 6-mer siRNA
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
    
data_siRNA_senseNoflank=df_raw.values[:,1]
data_6mer=[seq2kmer(i,6) for i in data_siRNA_senseNoflank]
print(data_6mer[0])
label=list(df_raw.values[:,4])
df=pd.DataFrame({'scores':label ,'text':data_6mer})
print(df)

###Construct BERTbased model

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
        
###DNABERT-4 Layer with attention and last 3 layers

class siRNABertRegressor(nn.Module):

    def __init__(self, dropout=0.1):

        super(siRNABertRegressor, self).__init__()
        
        #self.bert = BertModel.from_pretrained(r'.\DNABERTbased\3mer')
        #self.bert = BertModel.from_pretrained(r'.\DNABERTbased\4mer')
        #self.bert = BertModel.from_pretrained(r'.\DNABERTbased\5mer')
        config=BertConfig.from_pretrained(r'.\DNABERTbased\6mer', output_attentions=True)
        self.bert = BertModel.from_pretrained(r'.\DNABERTbased\6mer',config=config)      
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 128)
        #self.linear1 = nn.Linear(768, 256)
        self.relu1 = nn.ReLU()
        #self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        #self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        #self.sigmoid3 = nn.Sigmoid()
        self.linear4 = nn.Linear(64, 1)
        #self.relu4 = nn.ReLU()
        self.sigmoid4 = nn.Sigmoid()
        #self.sigmoid = nn.Sigmoid()
    def forward(self, input_id, mask):

        output = self.bert(input_ids= input_id, attention_mask=mask)
        #label_output = output[1]
        attention=output[-1]
        #num=len(output)
        label_output=output[0][:,0,:]
        dropout_output = self.dropout(label_output)
        linear_output1 = self.linear1(dropout_output)
        reluoutput1=self.relu1(linear_output1)
        #sigmoid1=self.sigmoid1(linear_output1)
        linear_output2=self.linear2(reluoutput1)
        #linear_output2=self.linear2(sigmoid1)
        reluoutput2=self.relu2(linear_output2)
        #sigmoid2=self.sigmoid2(linear_output2)
        linear_output3=self.linear3(reluoutput2)
        #linear_output3=self.linear3(sigmoid2)
        reluoutput3=self.relu3(linear_output3)
        #sigmoid3=self.sigmoid3(linear_output3)
        linear_output4=self.linear4(reluoutput3)
        #linear_output4=self.linear4(sigmoid3)
        #final_layer = self.relu3(linear_output4)
        final_layer = self.sigmoid4(linear_output4)

        return final_layer,attention,reluoutput1,reluoutput2,reluoutput3

model = siRNABertRegressor()

'''
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
model.apply(weight_init)
'''
classify=0.7
def evaluation(ytrue,pred): 
    out=np.vstack((pred,ytrue)).T
    dat=pd.DataFrame(out)
    Pearson=dat.corr().iloc[0,1]
    Spearman=dat.corr('spearman').iloc[0,1]
    return Pearson,Spearman

###Train, validate and test

np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])
print(len(df_train),len(df_val), len(df_test))
epochs=30
learning_rate=5e-5
train_data=df_train
val_data=df_val
test_data=df_test
train, val,test = Dataset(train_data), Dataset(val_data), Dataset(test_data)
train_dataloader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=10)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=10)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

seed = 2
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
#optimizer = Adam(model.parameters(), lr= learning_rate)
optimizer = Adam(model.parameters(), lr= learning_rate, weight_decay=1e-2)
if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()
        
loss_train=[]
loss_val=[]
loss_test=[]
cor_train=[]
cor_val=[]
cor_test=[]
for epoch_num in range(epochs):

        #total_acc_train = 0
        Out_train=[]
        Label_train=[]
        total_correlation_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            epoch_loss=[]
            #train_label = train_label.to(device).long()
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output,attention,last3,last2,last1 = model(input_id, mask)

            batch_loss = criterion(output.squeeze(), train_label.squeeze())
            total_loss_train += batch_loss.item()
            #total_loss_train.append(batch_loss.item())
            Out_train.extend(output)
            #Out_train.append(output)
            Label_train.extend(train_label)

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        l_train=[i.detach().cpu().numpy().tolist() for i in Label_train]
        p_train=[i.detach().cpu().numpy().tolist()[0] for i in Out_train]
        #p_train=[i.detach().cpu().numpy().tolist()[0][0] for i in Out_train]
        correlation_train=evaluation(l_train,p_train)[0]
        
        #total_acc_val = 0
        Out_val=[]
        Out_test=[]
        Label_val=[]
        Label_test=[]
        total_correlation_val=0
        total_correlation_test=0
        total_loss_val = 0
        total_loss_test = 0
        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output,attention,last3,last2,last1 = model(input_id, mask)
                batch_loss = criterion(output.squeeze(), val_label.squeeze())
                total_loss_val += batch_loss.item()
                #total_loss_val.append(batch_loss.item())
                Out_val.extend(output)
                Label_val.extend(val_label)
            l_val=[i.detach().cpu().numpy().tolist() for i in Label_val]
            p_val=[i.detach().cpu().numpy().tolist()[0] for i in Out_val]
            correlation_val=evaluation(l_val,p_val)[0]

            
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)
                output,attention,last3,last2,last1 = model(input_id, mask)
                batch_loss = criterion(output.squeeze(), test_label.squeeze())
                total_loss_test += batch_loss.item()
                #total_loss_test.append(batch_loss.item())
                Out_test.extend(output)
                Label_test.extend(test_label)
            l_test=[i.detach().cpu().numpy().tolist() for i in Label_test]
            p_test=[i.detach().cpu().numpy().tolist()[0] for i in Out_test]
            correlation_test=evaluation(l_test,p_test)[0]         
        loss_train.append(round(mean(total_loss_train),3))
        loss_val.append(round(mean(total_loss_val),3))
        loss_test.append(round(mean(total_loss_test),3))
        cor_train.append(round(correlation_train,3))
        cor_val.append(round(correlation_val,3))
        cor_test.append(round(correlation_test,3))
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train : .3f} \
            | Train Pearson\'s Correlation: {correlation_train: .3f} \
            | Val Loss: {total_loss_val : .3f} \
            | Val Pearson\'s Correlation: {correlation_val: .3f}\
            | Test Loss: {total_loss_test : .3f} \
            | Test Pearson\'s Correlation: {correlation_test: .3f}'
        )