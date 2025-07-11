
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
    
    vector_predict=vector.transform([value])
    model_predict_valuer=model.predict(vector_predict)
    return (f'News is:',int(model_predict_valuer[0]))