from flask import Flask,request,jsonify
import pandas as pd 
import joblib
import json
import numpy as np

app = Flask(__name__)
@app.route("/",methods=['GET','POST'])
def mushrooms_classifing():
    if request.method == "POST":
        model = joblib.load("mushrooms_cls.pkl")
        pca = joblib.load("pca.pkl")
        raw = pd.read_csv("mushrooms.csv")
        raw_X = raw.drop("class")
        raw_dummies = pd.get_dummies(raw_X)
        json_data = request.json
        #print("received data:",json_data)
        #json_dict = json.loads(json_data)
        #json_int = {k:int(v) for k,v in json_dict.items()}
        df = pd.DataFrame(json_data)
        df_dummies = pd.get_dummies(df).reindex(columns=raw_dummies.columns,fill_value=0)
        pca_data = pca.transform(df_dummies)
        prediction = model.predict(pca_data)
        
        res_dict = {}
        y_proba = model.predict_proba(pca_data)
        for i in range(len(y_proba)):
            if any(y_proba[i] > 0.8):
                i_proba = y_proba[i][y_proba[i]>0.8]
                i_cls = model.classes_[y_proba[i]==i_proba]
                print(f"{i}th input data's prediction is {i_proba} ,belong to class {i_cls},adding to training")
                model.partial_fit(np.atleast_2d(pca_data[i]),np.atleast_2d(i_cls))
                res_dict[i] = {"prediction":prediction[i].item(),"probability":i_proba.item(),"back":"added to training"}
            else:
                res_dict[i] = {"prediction":prediction[i].item(),"probability":max(y_proba[i]).item(),"back":"not added to training"}
        #print("prediction:",prediction)
        return jsonify(res_dict)
    else:
        return jsonify({"waiting":"data"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4096)
