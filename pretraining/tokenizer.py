import re
import urllib.request
from collections import defaultdict

URL=("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
filepath='the-verdict.txt'
urllib.request.urlretrieve(URL, filepath)

class Encoder():
    def __init__(self, href):
        self.href= href
        self.rawtext=""
        self.mapping=[]         #mapping from unique tokens->id
        self.inv_mapping = []   #mapping from id->unique tokens

    def __str__(self):
        if not self.rawtext:
            with open('the-verdict.txt') as file:
                self.rawtext=file.read()
                rt=re.split(r'([:;,.?!"()]|--|\s)', self.rawtext)
                rtt = final=[x for x in rt if x!='']
                result = []
                for char in rtt:
                    if char in self.mapping:
                        result.append({char: self.mapping[char]})
                    else:
                        result.append({char: None})
                return f"Token to ID mapping: {result}"
            
    def encode(self):
        if not self.mapping:
            with open(self.href) as file:
                rawtext = file.read()
                result=re.split(r'([:;,.?!"()]|--|\s)', rawtext)
                rstripped=map(lambda x: x.strip(), result)
                final=[x for x in rstripped if x!='']
                dictionary={word: idx for idx, word in enumerate(set(final))}
                self.mapping = dictionary
                

    def decode(self):
        if not self.inv_mapping:
            self.inv_mapping = {self.mapping[key]: key for key in self.mapping}

e = Encoder(filepath)
e.encode()
e.decode()
print(str(e))