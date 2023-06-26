from os.path import dirname, abspath, join
import pickle
from transformers import AutoTokenizer
from scipy.special import softmax

tokenizer = None

def init():
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    
    current_dir = dirname(abspath(__file__))
    model_path = abspath(join(current_dir, '..', '..', 'models', 'twt_roberta_model.pkl')).replace('\\\\', '/')
    
    return pickle.load(open(model_path, 'rb'))

def prepare(text):
    return tokenizer(text, return_tensors='pt')


def predict_sentiment(model, text):
    output = model(**text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores.argmax()