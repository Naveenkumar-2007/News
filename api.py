from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import nltk

# Ensure required NLTK data is available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

app = FastAPI()

with open("models/vector.pkl", "rb") as f:
    vector = pickle.load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)


class BaseModelRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "hey welcome"}


@app.post("/predict")
def predict_re(item: BaseModelRequest):
    try:
        value = item.text

        # remove special characters
        value = re.sub("[^a-zA-Z0-9 ]", "", value)

        # remove stopwords
        try:
            sw = set(stopwords.words("english"))
            value = " ".join([word for word in value.split() if word.lower() not in sw])
        except LookupError:
            return {"error": "NLTK stopwords not found. Please ensure stopwords are installed."}

        # remove HTML tags
        value = BeautifulSoup(value, "lxml").get_text()

        # remove links
        value = re.sub(
            r"(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
            "",
            str(value),
        )

        # remove extra spaces
        value = " ".join(value.split())

        # apply lemmatization
        try:
            value = " ".join(WordNetLemmatizer().lemmatize(i) for i in value.split())
        except LookupError:
            return {"error": "NLTK wordnet not found. Please ensure wordnet is installed."}

        # vectorize and predict
        vector_predict = vector.transform([value])
        model_predict_valuer = model.predict(vector_predict)

        if model_predict_valuer == 0:
            return {"prediction": "This News is Fake ⚠️☠️🚨"}
        else:
            return {"prediction": "This News is Real 😉"}

    except Exception as e:
        return {"error": str(e)}
