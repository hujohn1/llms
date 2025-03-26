import re
import torch
import urllib.request
import tiktoken
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

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

tokenizer = Encoder(filepath)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
text2 = "Hello, do you like tea?"
tokenizer.tokenize()
ids = tokenizer.encode(text2)
print(ids)

regen = tokenizer.decode(ids)
print(regen)


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
dataset = GPTDataset(rawtext, ttokenizer, max_length=256, stride=128)
trainloader = DataLoader(
    dataset=dataset,
    batch_size= 256, 
    shuffle= True, 
    num_workers=0
)
data_iter=iter(trainloader)
batch_one=next(data_iter)
print(batch_one)
