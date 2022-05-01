from os.path import exists
import pandas as pd
from pathlib import Path
import json

df = pd.read_json("docs.jsonl", lines = True)
c = 0
ch = 0
che = 0
df['body_text'] = ''
for i in df['cord19_id'].tolist():
    che+=1
    try:
        s = i + ".json"
    except:
        continue
    text = ""
    ch += 1
    path = "/home/sraval/document_parses/pdf_json/"+s
    if(exists("/home/sraval/document_parses/pdf_json/"+s)):
        c+=1
        f = open(path)
        data = json.load(f)
        for j in data["body_text"]:
            text+=j["text"]
        df.loc[df.cord19_id == i, 'body_text'] = text
df.to_json('temp.jsonl', orient='records', lines=True)
print(c)
print(ch)
print(che)
