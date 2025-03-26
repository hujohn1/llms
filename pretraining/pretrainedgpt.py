import re
import math
import torch
import torch.nn as nn
import urllib.request
import tiktoken
from colorama import Fore, Back, Style
import numpy as np
from torch.utils.data import Dataset, DataLoader

SEED = 123
URL=("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
filepath='the-verdict.txt'
urllib.request.urlretrieve(URL, filepath)

class Encoder():
    def __init__(self, href):
        self.href= href
        self.rawtext=""
        self.mapping = []
        self.inv_mapping = []

    def __str__(self):
        return self.rawtext
    
    def tokenize(self):
        if not self.mapping:
            with open(self.href) as file:
                rawtext = file.read()
                result=re.split(r'([:;,.?!"()]|--|\s)', rawtext)
                rstripped=map(lambda x: x.strip(), result)
                final=[x for x in rstripped if x!='']
                dictionary={word: idx for idx, word in enumerate(set(final))}
                self.mapping = dictionary
                result=[self.mapping[tk] for tk in final if tk in self.mapping]
                self.inv_mapping = {self.mapping[key]: key for key in self.mapping}
                return result

    def encode(self, text):
        result=re.split(r'([:;,.?!"()]|--|\s)', text)
        rstripped=map(lambda x: x.strip(), result)
        final=[x for x in rstripped if x!='']
        result=[self.mapping[tk] for tk in final if tk in self.mapping]
        return result

    def decode(self, ids):
        result = [self.inv_mapping[tk] for tk in ids if tk in self.inv_mapping]
        return result

    #adding special characters <|unk|>, <|endoftext|>, <PAD>, <MASK>, <CLS>

#tokenizer = Encoder(filepath)
#text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
#text2 = "Hello, do you like tea?"
#tokenizer.tokenize()
#ids = tokenizer.encode(text2)
#print(ids)
#regen = tokenizer.decode(ids)
#print(regen)


#Byte Pair Encoding
enc = tiktoken.get_encoding("o200k_base")
print(enc.decode(enc.encode("Hello World")))

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.inputs=[]
        self.preds=[]
        enc=tokenizer.encode(text)
        print(f'Encoding length {len(enc)}')

        for i in range(0, len(enc)-max_length, stride):
            self.inputs.append(torch.tensor(enc[i: i+max_length]))
            self.preds.append(torch.tensor(enc[i+1: i+1+max_length]))

    def __getitem__(self, index):
        return self.inputs[index], self.preds[index]

    def __len__(self):
        return len(self.inputs)

with open('the-verdict.txt') as file:
    rawtext = file.read()

ttokenizer=tiktoken.get_encoding("gpt2")

def createDataLoader(rawtext, ttokenizer, batch_size, max_length, stride):
    dataset = GPTDataset(rawtext, ttokenizer, max_length, stride)
    trainloader = DataLoader(
        dataset=dataset,
        batch_size= batch_size, 
        shuffle= False, 
        num_workers=0, 
        drop_last=True
    )
    return trainloader

max_length=4
dataloader = createDataLoader(rawtext, ttokenizer, 8, 4, 4)
data_iter=iter(dataloader)
batch_one_inputs, batch_one_preds=next(data_iter)
print(batch_one_inputs)

torch.manual_seed(SEED)
vocab_size = 50527
output_dim = 256

tk_embedding_layer=torch.nn.Embedding(vocab_size, output_dim)
#initial weights
print(tk_embedding_layer.weight)
#token level embeddings
tk_embeddings = tk_embedding_layer(batch_one_inputs)
print(tk_embeddings.shape)

#Positional embeddings
pos_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

#Input embeddings
input_embeddings = pos_embeddings + tk_embeddings
print(input_embeddings.shape)


word="Your journey starts with one step"
emb=ttokenizer.encode(word)
print(emb)

#Simplified self-attention
#Context vector = attention weight * each element
#attention weight = normalized attention score A_ij=attention between xi and xj
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

attn_scores = inputs @ inputs.T
print(attn_scores)
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

content_vecs = attn_weights @ inputs
print(content_vecs)

torch.manual_seed(123)
class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SelfAttention, self).__init__()
        shape = (dim_in, dim_out)
        self.M_k = nn.Parameter(torch.rand(shape))
        self.M_q = nn.Parameter(torch.rand(shape))
        self.M_v = nn.Parameter(torch.rand(shape))
    
    def forward(self, X):
        queries = X @ self.M_q
        keys = X @ self.M_k
        values = X @ self.M_v
        normed_head=torch.softmax((queries @ keys.T)/keys.shape[-1]**0.5, dim=-1)
        print(f"NORMED HEAD \n{normed_head}")
        mask = torch.tril(torch.ones((normed_head.shape[0], normed_head.shape[1])))
        masked_head = mask * normed_head
        row_sums = masked_head.sum(dim=-1, keepdim=True)
        masked_simple_norm = masked_head / row_sums

        print(Fore.RED + f"MASKED HEAD"+Style.RESET_ALL+f"\n{masked_head}")   
        print(Fore.GREEN + f"NORMED MASKED HEAD" +Style.RESET_ALL+f"\n{masked_simple_norm}")
        
        content_vec = (masked_simple_norm) @ values
        return content_vec
    
sav = SelfAttention(3, 2)
print(sav(inputs))

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        assert dim_out % num_heads ==0
        self.heads = nn.ModuleList([SelfAttention(dim_in, dim_out) for _ in range(num_heads)])
    def forward(self, X):
        return torch.cat([head(x) for head in self.heads], dim=-1)