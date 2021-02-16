from sklearn.neural_network import MLPClassifier
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report
import torch
from sklearn.model_selection import train_test_split
import re
import pickle


def filter(text):

    final_text = ''
    final_text = text.replace("<br />"," ")
    final_text = text.replace("  "," ")

    return final_text


tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-128k-uncased").to()

try:
    def feature_extraction(text):
        x = tokenizer.encode(filter(text))
        with torch.no_grad():
            x, _ = bert(torch.stack([torch.tensor(x)]).to())
            return list(x[0][0].cpu().numpy())
except Exception as e:
    print(e)


with open("/home/felix/PycharmProjects/turkishnlp/csvjson.json", 'r') as f:
    data = json.load(f)

x = []
y = []

for i in data:
    try:
        x.append(feature_extraction(i["Görüş"]))
        if (i["Durum"] == "Tarafsız"):
            y.append(1)
        elif (i["Durum"] == "Olumsuz"):
            y.append(0)
        elif (i["Durum"] == "Olumlu"):
            y.append(2)
    except Exception as e:
        print(e)
        continue

x_train, x_test, y_train, y_test = train_test_split(x, y)


model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=10000, activation='tanh', solver='adam', alpha=1e-5, learning_rate='constant',
                      verbose=1, early_stopping=True)
model.fit(x_train, y_train)
y_true, y_pred = y_test, model.predict(x_test)
print(classification_report(y_true, y_pred))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))