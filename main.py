# Load libraries
from flask import Flask, render_template, request, jsonify
import pandas as pd
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.models import load_model
import spacy
#from spacy.pipeline import Sentencizer
#import spacy.cli
#spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")
#sentencizer = Sentencizer()
#nlp.add_pipe(sentencizer, first=True)

app = Flask(__name__)
app.config['DEBUG'] = True


# load the model, and pass in the custom metric function

model = load_model('baseline-model.h5')
emoji_dict = {0: '❤️', 1: '😍', 2: '😂', 3: '💕', 4: '🔥', 5: '😊', 6: '😎', 7: '✨', 8: '💙', 9: '😘', 10: '📷', 11: '🇺🇸', 12: '☀', 13: '💜', 14: '😉', 15: '💯', 16: '😁', 17: '🎄', 18: '📸', 19: '😜' }
@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict(): 
    message = request.form
    prediction_text = message['predictionText']

    df = pd.DataFrame([prediction_text], index=[0], columns = ['text'])
    X_test_indices = sentences_to_indices(df['text'], 50)
    emoji_index = np.argsort(model.predict(X_test_indices), axis=1)[:,-3:]
    pred = []
    for emoji in emoji_index[0]:
       
        pred = np.append(pred, emoji_dict[emoji])
    
    response = {
        'emoji_1': pred[2],
        'emoji_2': pred[1],
        'emoji_3': pred[0]
    }
    return jsonify(response)


def sentences_to_indices(X, max_len):

    train_ex = X.shape[0]                         
    X_indices = np.zeros((train_ex, max_len))

    for i in range(train_ex):                               
        
        j = 0
        entrada = nlp(str(X[i]))

        for token in entrada:
          try:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
              try:
                X_indices[i, j] =  vector_id
                j += 1
              except IndexError:
                pass
          except KeyError:
            pass 

    return X_indices



if __name__ == "__main__":
    app.run(host='localhost', port=7000)
    #run