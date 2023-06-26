from os.path import dirname, abspath, join
import joblib
import pickle

global cv

def init():
    global cv
    
    current_dir = dirname(abspath(__file__))
    bow_path = abspath(join(current_dir, '..', '..', 'data', 'interim', 'c1_BoW_Sentiment_Model.pkl')).replace('\\\\', '/')
    model_path = abspath(join(current_dir, '..', '..', 'models', 'random_forest_model.joblib')).replace('\\\\', '/')
    
    cv = pickle.load(open(bow_path, "rb"))
    return joblib.load(model_path)


def prepare(text):
    processed_input = cv.transform([text]).toarray()[0]
    return [processed_input]


def predict_sentiment(model, text):
    return model.predict(text)[0]