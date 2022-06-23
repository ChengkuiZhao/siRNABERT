import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from transformers import BertModel, BertConfig,BertTokenizer

###reynold Test data
reynold_datapath=r'data\testdata\reynold.csv'
reynold_raw=pd.read_csv(reynold_datapath,sep=',')
print(reynold_raw.shape[0])
###katoh Test data
katoh_datapath=r'data\testdata\katoh.csv'
katoh_raw=pd.read_csv(katoh_datapath,sep=',')
print(katoh_raw.shape[0])
###Test data
data_test_raw=pd.concat([reynold_raw,katoh_raw])
print(data_test_raw.head)
print(data_test_raw.shape[0])

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
    
data_test_text=data_test_raw.values[:,1]
data_test_6mer=[seq2kmer(i,6) for i in data_test_text]
print(data_test_6mer[0])
data_test_label=list(data_test_raw.values[:,4])
data_test=pd.DataFrame({'scores':data_test_label ,'text':data_test_6mer})
print(data_test)

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

###Load model and test

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
test_dataloader=torch.utils.data.DataLoader(Dataset(data_test),batch_size=1000)
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

        y_test=[round(i,4) for i in test_label.tolist()]
        predictions=[i[0] for i in test_output.tolist()]
        correlation_test=evaluate(y_test,predictions)
                 
print(f'Test Correlation: {correlation_test} ')

pred_renold=pd.read_csv(r'.\data\OthersPredictions\Predictions_renold.csv').drop(['Unnamed: 0'],axis=1)
pred_katoh=pd.read_csv(r'.\data\OthersPredictions\Predictions_katoh.csv').drop(['Unnamed: 0'],axis=1)

#AUC_pred_renold=[evaluate(pred_renold['Efficiency']/100,pred_renold[i])[2] for i in pred_renold.columns[:-1]]
#AUC_pred_katoh=[evaluate(pred_katoh['Efficiency']/100,pred_katoh[i])[2] for i in pred_katoh.columns[:-1]]

y_others=pd.concat([pred_renold['Efficiency'],pred_katoh['Efficiency']])
pred_iScore=pd.concat([pred_renold['i.Score'],pred_katoh['i.Score']])
pred_Thermo21=pd.concat([pred_renold['Thermo21'],pred_katoh['Thermo21']])
pred_DSIR=pd.concat([pred_renold['DSIR'],pred_katoh['DSIR']])
pred_Biopred=pd.concat([pred_renold['s.Biopredsi'],pred_katoh['s.Biopredsi']])

Evaluate_iScore=evaluate(y_others,pred_iScore)
Evaluate_Thermo21=evaluate(y_others,pred_Thermo21)
Evaluate_DSIR=evaluate(y_others,pred_DSIR)
Evaluate_Biopred=evaluate(y_others,pred_Biopred)
classify=0.7
y=[]
y.append([1 if i>=classify else 0 for i in y_test])
y.append([1 if i>=classify else 0 for i in y_others])
y.append([1 if i>=classify else 0 for i in y_others])
y.append([1 if i>=classify else 0 for i in y_others])
y.append([1 if i>=classify else 0 for i in y_others])
pred_SiRNABERT=predictions
Prediction=[]
Prediction.append(pred_SiRNABERT)
Prediction.append(pred_iScore)
Prediction.append(pred_Thermo21)
Prediction.append(pred_DSIR)
Prediction.append(pred_Biopred)
print(correlation_test)
print(Evaluate_iScore)
print(Evaluate_Thermo21)
print(Evaluate_DSIR)
print(Evaluate_Biopred)
Evaluate_Pearson=[correlation_test[0],Evaluate_iScore[0],Evaluate_Thermo21[0],Evaluate_DSIR[0],Evaluate_Biopred[0]]
Evaluate_Spearman=[correlation_test[1],Evaluate_iScore[1],Evaluate_Thermo21[1],Evaluate_DSIR[1],Evaluate_Biopred[1]]

##Plot correlation
def addlabels2(x,y):
    for i in range(len(x)):
        plt.text(i-0.25,y[i]+0.003,round(y[i],3),fontsize=12)
def addlabels1(x,y):
    for i in range(len(x)):
        plt.text(i+0.05,y[i]+0.003,round(y[i],3),fontsize=12)        
algorithm=['SiRNABERT','i-Score','Thermo21','DSIR','Biopredsi']
plt.figure(dpi=300,figsize=(15,8))
x_width=range(len(Evaluate_Pearson))
x1_width=[i-0.15 for i in x_width]
x2_width=[i+0.15 for i in x_width]

plt.bar(x1_width,Evaluate_Spearman,lw=1,fc='tomato',width=0.3,label='Spearman')
plt.bar(x2_width,Evaluate_Pearson,lw=1,fc='c',width=0.3,label='Pearson')
plt.xlabel('Algorithm',fontsize=20)
plt.ylabel('Correlation',fontsize=20)
plt.xticks(range(0,5),algorithm)
plt.ylim(0.5,0.65)
plt.legend(loc="upper right",fontsize=15)
plt.tick_params(labelsize=15)
addlabels1(x_width,Evaluate_Pearson)
addlabels2(x2_width,Evaluate_Spearman)
plt.show()

###Plot ROC
from numpy import interp
from sklearn import metrics
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
plt.figure(dpi=300,figsize=(15,12))
algorithm=['SiRNABERT','i-Score','Thermo21','DSIR','Biopredsi']
for i in range(0,5) :
    y_val=y[i]
    fpr,tpr,thresholds=metrics.roc_curve(y_val,Prediction[i],pos_label=1)
    mean_tpr+=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,label=algorithm[i]+' (area=%0.3f)' % (roc_auc),linewidth=2.5)

plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label='luck')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('Receiver operating characteristic ',fontsize=30)
plt.legend(loc="lower right",fontsize=20)
plt.tick_params(labelsize=15)
plt.show()  

###Plot PRC
mean_recall = np.linspace(0, 1, 100)
plt.figure(dpi=300,figsize=(15,12))
algorithm=['SiRNABERT','i-Score','Thermo21','DSIR','Biopredsi']
for i in range(0,5) :
    y_val=y[i]
    precision,recall,thresholds=metrics.precision_recall_curve(y_val,Prediction[i],pos_label=1)
    prc_auc=metrics.auc(recall,precision)
    plt.plot(recall,precision,label=algorithm[i]+' (area=%0.3f)' % (prc_auc),linewidth=2.5)

plt.xlim([0, 1])
plt.ylim([0.3, 1])
plt.xlabel('Recall',fontsize=20)
plt.ylabel('Precision',fontsize=20)
plt.title('Precision Recall Curve',fontsize=30)
plt.legend(loc="lower left",fontsize=20)
plt.tick_params(labelsize=15)
plt.show()  

###Model attention visualization

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def get_attention_dna(model, tokenizer, sentence_a, start, end,device):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b=None, return_tensors='pt',max_length = 16, add_special_tokens=True).to(device)
    mask = inputs['attention_mask'].to(device)
    input_ids = inputs['input_ids'].to(device)
    attention = model(input_ids,mask)[1]
    input_id_list = input_ids[0] # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 
    attn = format_attention(attention)
    attn_score = []
    for i in range(1, len(tokens)-1):
        attn_score.append(float(attn[start:end+1,:,0,i].sum()))
    return attn_score
    
def get_real_score(attention_scores, kmer, metric):
    counts = np.zeros([len(attention_scores)+kmer-1])
    real_scores = np.zeros([len(attention_scores)+kmer-1])

    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i+j] += 1.0
                real_scores[i+j] += score

        real_scores = real_scores/counts
    else:
        pass

    return real_scores
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--kmer",
    default=6,
    type=int,
    help="K-mer",
)
parser.add_argument(
    "--model_path",
    default=r".\DNABERTbased\6mer",
    type=str,
    help="The path of the finetuned model",
)
parser.add_argument(
    "--start_layer",
    default=0,
    type=int,
    help="Which layer to start",
)
parser.add_argument(
    "--end_layer",
    default=11,
    type=int,
    help="which layer to end",
)
parser.add_argument(
    "--metric",
    default="mean",
    type=str,
    help="the metric used for integrate predicted kmer result to real result",
)
parser.add_argument(
    "--sequence",
    default= "ATGTTGCCGTCCTCCTTGAAG",
    type=str,
    help="the sequence for visualize",
)
args = parser.parse_args(args=[])

##eff=0.96
SEQUENCE='CTTGACTGGCGACGTAATCCA'
# load model and calculate attention
tokenizer_name = r'.\DNABERTbased\6mer'
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")       
tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
raw_sentence=SEQUENCE
sentence_a = seq2kmer(raw_sentence, args.kmer)
tokens = sentence_a.split()

attention = get_attention_dna(model, tokenizer, sentence_a, start=args.start_layer, end=args.end_layer,device=device)
attention_scores = np.array(attention).reshape(np.array(attention).shape[0],1)
real_scores = get_real_score(attention_scores, args.kmer, args.metric)
scores = pd.DataFrame(real_scores.reshape(1, real_scores.shape[0]))

sns.set()
ax = sns.heatmap(scores.transpose(), cmap='YlGnBu')
plt.show()

Testset=list(data_test_raw['siRNA'])
Scores=[]
for SEQUENCE in Testset:
    raw_sentence=SEQUENCE
    sentence_a = seq2kmer(raw_sentence, args.kmer)
    tokens = sentence_a.split()
    attention = get_attention_dna(model, tokenizer, sentence_a, start=args.start_layer, end=args.end_layer,device=device)
    attention_scores = np.array(attention).reshape(np.array(attention).shape[0],1)
    real_scores = get_real_score(attention_scores, args.kmer, args.metric)
    scores = real_scores.reshape(1, real_scores.shape[0]).tolist()[0]
    Scores.append(scores) 
    
###Visualize last 3 layers clustering and TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

last3_new=[i.cpu().numpy().tolist() for i in last3]
last2_new=[i.cpu().numpy().tolist() for i in last2]
last1_new=[i.cpu().numpy().tolist() for i in last1]

classify=0.7
x = last3_new
y = [1 if i>=classify else 0 for i in y_test]
tsne = TSNE(n_components=2, verbose=1,perplexity=30, random_state=0)
z = tsne.fit_transform(x)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Test data T-SNE projection") 

x = last2_new
y = [1 if i>=classify else 0 for i in y_test]
tsne = TSNE(n_components=2, verbose=1,perplexity=30, random_state=0)
z = tsne.fit_transform(x)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Test data T-SNE projection") 

x = last1_new
y = [1 if i>=classify else 0 for i in y_test]
tsne = TSNE(n_components=2, verbose=1,perplexity=30, random_state=0)
z = tsne.fit_transform(x)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Test data T-SNE projection") 
    
