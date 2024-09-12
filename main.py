#codage de l'API:pip install fastapi
#installation de fastApi et uvicorn(serveur):pip install uvicorn

#librairies
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris

#jeud de données à remettre
iris=load_iris
#chargement du modèle
loaded_model=load('logreg.joblib')
#création d'une nouvelle instance fastapi
app=FastAPI()

#définir un objet (une classe) pour réaliser des requêtes
class request_body(BaseModel):
    #features
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_wdth : float
    
#définition du scéhma du point de terminaison (API)
@app.post("/predict")#local : http://127.0.0.1:8000/predict


#définition de la fonction de prédiction
def predict(data : request_body) :
    #nouvelles données sur lesquelles on fait la prédiction
    new_data=[[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_wdth
    ]]
    
    #prediction à une seule valeur
    class_idx=loaded_model.predict(new_data)[0]
    
    #je retourne le nom de l'espèce iris
    return {'class':iris.target_names[class_idx]}