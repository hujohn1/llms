import re
import urllib.request

URL=("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
filepath='the-verdict.txt'
urllib.request.urlretrieve(URL, filepath)

try:
    with open('the-verdict.txt') as file:
        rawtext = file.read()
        print(rawtext[:99])
        result=re.split(r'([:;,.?!"()]|\s)', rawtext[:99])
        rstripped=map(lambda x: x.strip(), result)
        final=[x for x in rstripped if x!='']
        print(final)

except:
    print("error")