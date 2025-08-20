
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import pandas as pd
app=FastAPI()
with open('models/vector.pkl','rb') as f:
    vector=pickle.load(f)
with open('models/model.pkl','rb') as f:
    model=pickle.load(f)

class base_model(BaseModel):
    text:str
    
    


@app.get('/')
def root():
    return 'hey welcome'

@app.post('/predict')
def predict_re(item:base_model):
    value=item.text
    ##remove speacial charectors
    value=re.sub('[^a-z A-Z 0-9]','',value)
    ##remove stopwords
    value=" ".join([ wor for wor in value.split() if wor not in stopwords.words('english')])
    
    ##remove html tags
    value=BeautifulSoup(value,'lxml').get_text()
    #remove links
    value=re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','',str(value))
    ##remove spaces
    value=" ".join(value.split())
    ##apply wordlematizing
   
    value=" ".join(WordNetLemmatizer().lemmatize(i) for i in value.split())
    


    vector_predict=vector.transform([value])
    model_predict_valuer=model.predict(vector_predict)
    #return (f'News is:',int(model_predict_valuer[0]))
    if model_predict_valuer==0:
        return 'This News is Fack ⚠️☠️🚨'
    else:
        return 'This News is Real 😉'