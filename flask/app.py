import math
import pandas as pd 
import numpy as np 
import flask
import pickle as pk
from flask import Flask, render_template, request

app=Flask(__name__)

def init():
    global data_encoder, decisionTreeRegr

    print("initializing... ") 
    data_encoder = pk.load(open("label_encoder_map.pkl", "rb"))
    decisionTreeRegr = pk.load(open("dec_tree_regr.pkl", "rb")) 

    print("initialized") 


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/result', methods = ['POST'])
def predict():
    tagValuePairs = request.form.to_dict()
    if (tagValuePairs['fuelType'] == "gasoline"):
        tagValuePairs['fuelType'] = "petrol"
    print(tagValuePairs)

    # incoming request
    to_predict_list = tagValuePairs
    to_predict_list = list(to_predict_list.values())
    print(to_predict_list)

    # build the structure to run this data through the model
    to_predict = np.array(to_predict_list).reshape(1, 6)
    print(to_predict)

    df = pd.DataFrame(to_predict, columns=['Brand', 'model', 'Year', 'kmDriven', 'Transmission', 'FuelType'])

    # everything lowercase
    df = df.applymap(lambda s: s.lower() if isinstance(s, str) else s)

    #  numeric columns --> type(fingers crossed)
    df['Year'] = df['Year'].astype(int)
    df['kmDriven'] = df['kmDriven'].astype(float)
    print(df)
    
    # encode categorical columns
    try:
        df.replace(data_encoder, inplace=True)
        y_pred = decisionTreeRegr.predict(df)
        print("X=%s, Predicted=%s" % (df.to_numpy(), y_pred[0]))
        
        dollars = round(y_pred[0]/74.5, 2)  # exchange rate, rupee to dollar
        return render_template('result.html', prediction=("$"+str(dollars)))
    except:
        return render_template('result.html', prediction="We don't have info on your car :(")
    


if __name__ == '__main__':
    init()
    app.run(debug=True, port=9090)
   
