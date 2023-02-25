from flask import Flask
import json
import requests
import pandas as pd
import io
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)

@app.route('/')
def home():
    return 'App working! '

@app.route('/pca')
def pca_data():
    url = "https://raw.githubusercontent.com/thota-sasanth/Accidents-Dashboard/main/visualization_projdata.csv"
    response = requests.get(url)
    df = pd.read_csv(io.BytesIO(response.content))
    df_forpca = df.drop(columns=['State', 'County','Mode_Weather_Condition','Mode_Severity'])
    df_forpca = (df_forpca - df_forpca.mean()) / df_forpca.std()
    pca = PCA()
    pca.fit(df_forpca)
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues.tolist()
    eigenvalues = json.dumps(eigenvalues)
    return eigenvalues




if __name__ == '__main__':
    app.run(port=8080, debug=True)


