
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from docx import Document
from nltk.tokenize import sent_tokenize
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

datafolder = 'Path-to-transcript'
outputfolder = 'path-to-output'

datalabel='1'

#########
# utility functions
##########
#########
# Parameters
###########

def read_docx_as_string(file_path):
    # Load the document
    doc = Document(file_path)

    # Extract all text from paragraphs and join into a single string
    full_text = '\n'.join([para.text for para in doc.paragraphs])
    full_text = full_text.replace(u'\xa0', u' ')

    # Remove non-ASCII (Unicode) characters by encoding to ASCII and ignoring errors
    # ascii_text = full_text.encode('ascii', 'ignore').decode('ascii')

    return full_text

#compute perplexity as negative log likelihood
def likelihood_of_sentences(sentences,model,tokenizer,max_length):
    nlls = []
    input_text=[tokenizer.encode(s, return_tensors="pt") for s in sentences]
    sentence_ends=np.cumsum([x.shape[1] for x in input_text])
    input_text=torch.cat(input_text,dim=1)
    #print(sentence_ends)
    print(input_text.shape)
    for i in tqdm(range(len(sentence_ends))):
        end_loc = sentence_ends[i]
        begin_loc = max(0,sentence_ends[i]-max_length)
        if i==0:
            trg_len=0
        else:
            trg_len = sentence_ends[i]-sentence_ends[i-1]  # the sentence to be evaluated
        input_ids=input_text[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 #don't include loss for all tokens except for the current evaluated sentence
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean())
    return nlls, ppl

########
# script
#########

currentSetup={'ClusterThreshold': .8,
                        'MinPerplexity': 2,
                        'PerplexityModel': "gpt2-medium",
                        'EmbeddingModel': 'all-mpnet-base-v2',
                        'sentimentModel': "cardiffnlp/twitter-roberta-base-sentiment",
                        'sentiment_max_length': 90,  # should not exceed hard limit of the model which is 512
                        'sentiment_max_sent': 4,
                        'cutoff_x': .5},
dThreshold=.8 #distance threshold for sentence clustering
model_name = "gpt2-medium"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
embedder = SentenceTransformer('all-mpnet-base-v2') #better performance
max_length = model.config.n_positions
minPerplex = 2 #perplexity threshold sentence binding for theme extraction


if Path(datafolder+datalabel+'.docx').exists():
    paragraph = read_docx_as_string(datafolder+datalabel+'.docx')
elif Path(datafolder+datalabel+'.txt').exists():
    with open(datafolder+datalabel+'.txt', 'r') as file:
        paragraph = file.read().replace('\n', '')
else:
    print('Cannot find transcripts for '+datalabel)

sentences=[]
for s in sent_tokenize(paragraph):
  sentences.append(s)

print('Data label: '+datalabel+'\n')
print(f'Number of of sentences: {len(sentences)}')

nlls,ppl=likelihood_of_sentences(sentences,model,tokenizer,max_length)
perplexityCon=[x.item() for x in nlls]
featureDF=pd.DataFrame({'text':sentences, 'perplexity':perplexityCon})

#process sentences, combine, if perplexity low
newsentences=[]
sentencecnt=[]
cnt=1
currentSen=sentences[0]
for i in range(len(featureDF)-1):
  if featureDF.loc[i+1,'perplexity']>minPerplex:
    newsentences.append(currentSen)
    sentencecnt.append(cnt)
    cnt=1
    currentSen=sentences[i+1]
    featureDF.loc[i+1:,'sentblock']=featureDF.loc[i+1:,'sentblock']+1
  else:
    currentSen=currentSen+' '+sentences[i+1]
    cnt=cnt+1

newsentences.append(currentSen) #newsentences are used for clustering later
sentencecnt.append(cnt)
print(f'Sentence filter by perplexity: before {len(sentences)}, after {len(newsentences)}\n')
print(sentencecnt)
print(featureDF.head())
print(len(featureDF))

#clustering to find topic
corpse=newsentences
corpus_embeddings = embedder.encode(newsentences)
clustering_model = AgglomerativeClustering(
    n_clusters=None, distance_threshold=dThreshold,metric='cosine', linkage='average'
)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []
    clustered_sentences[cluster_id].append(corpse[sentence_id])

print(f'number of sentences: {len(sentences):1d}')
print(f'number of clusters: {len(clustered_sentences):1d}')
num_transition=len(cluster_assignment)-sum(np.diff(cluster_assignment)==0)
print(f'number of transitions: {num_transition:1d}')

cluster_assign_bySentence=[]
for i in range(len(cluster_assignment)):
  cluster_assign_bySentence.extend([cluster_assignment[i]]*sentencecnt[i])

featureDF['cluster']=cluster_assign_bySentence
featureDF.to_csv(outputfolder+'features_'+datalabel+'.csv',index=False)
