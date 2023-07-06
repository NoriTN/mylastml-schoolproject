from flask import Flask, render_template, request
import joblib
import os
import pickle

MODEL_PATH = "/Users/franchouillard/Documents/GitHub/HousePriceKedro/data/06_models/regressor.pickle"
app = Flask(__name__)
def list_models(model_dir):
    if not os.path.exists(model_dir):
        print(f"Le dossier '{model_dir}' n'existe pas.")
        return []
    models = []
    for subdir in os.listdir(model_dir):
        subdir_path = os.path.join(model_dir, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            if 'regressor.pickle' in files:
                model_path = os.path.join(subdir_path, 'regressor.pickle')
                models.append(model_path)
    return models
def get_latest_model(models):
    sorted_models = sorted(models, key=lambda x: x.split("/")[-2], reverse=True)
    if sorted_models:
        latest_model = sorted_models[0]
        return latest_model

    return None
def load_model(model_path):
    model_pkl = pickle.load(open(model_path, 'rb'))
    return model_pkl
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        model = load_model(get_latest_model(list_models(MODEL_PATH)))
        lotarea = int(request.form['lotarea'])
        poolarea = int(request.form['poolarea'])
        prediction = model.predict([[lotarea, poolarea]])
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
