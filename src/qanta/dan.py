from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path
import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request
import time
from torch.utils.data import Dataset, DataLoader
from qanta import util
from qanta.dataset import QuizBowlDataset
from qanta import preprocess
import torch
from torch import nn
import nltk
from tqdm import tqdm
import numpy as np

MODEL_PATH = 'data/danmodel.pickle'
TORCH_MODEL_PATH = 'data/danmodel.pyt'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs

class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    def __init__(self, n_classes, vocab, glove_model, emb_dim=300,
                n_hidden_units=100, nn_dropout=.5):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout

        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        pretrained_weights = np.zeros((self.vocab_size, 300))

        for i, word in enumerate(vocab):
            try:
                pretrained_weights[i] = glove_model[word]
            except:
                pretrained_weights[i] = glove_model['unk']
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))


        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        #### modify the init function, you need to add necessary layer definition here
        #### note that linear1, linear2 are used for mlp layer
        self.act = nn.ELU()
        self._softmax = nn.Softmax()
        self.dropout = nn.Dropout(nn_dropout)
        #self.hidden = nn.Linear(n_hidden_units, n_hidden_units)
        self.classifier = nn.Sequential(self.linear1, self.act, self.dropout, self.linear2)


    def forward(self, input_text, text_len, is_prob=False, argmax = False):
        """
        Model forward pass

        Keyword arguments:
        input_text : vectorized question text 
        text_len : batch * 1, text length for each question
        in_prob: if True, output the softmax of last layer

        """
        #### write the forward funtion, the output is logits
        text_embed = self.embeddings(input_text)

        encoded = text_embed.sum(1)
        #encoded = encoded/text_embed.size(1)
        encoded = encoded/text_len.view(text_len.size(0),-1)
        logits = self.classifier(encoded)

        if is_prob:
            logits = self._softmax(logits)
        if argmax:
            logits = np.argmax(logits, axis = 1)

        return logits


class DanGuesser:
    def __init__(self, device='cpu'):
        self.dan_model = None
        self.i_to_class = None
        self.class_to_i = None
        self.voc = None
        self.ind2word = None
        self.word2ind = None
        self.device = device
        self.glove_model = None

    def train(self, training_data) -> None:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = DanGuesser()
            guesser.i_to_class = params['i_to_class']
            guesser.class_to_i = params['class_to_i']
            guesser.voc = params['voc']
            guesser.ind2word = params['ind2word']
            guesser.word2ind = params['word2ind']
            num_classes = params['num_classes']
            guesser.glove_model = params['glove_model']
            guesser.dan_model = DanModel(num_classes, guesser.voc, guesser.glove_model)
            guesser.dan_model.load_state_dict(torch.load(
                TORCH_MODEL_PATH))
            guesser.dan_model.eval()
        return guesser



 
class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser


def create_app(enable_batch=True):
    dan_guesser = DanGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(dan_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(dan_guesser, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the DAN model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    print('cuda status: {}'.format(torch.cuda.is_available()))
    print('Downloading punkt...')
    nltk.download('punkt')
    print('training DAN...')
    dan_guesser = DanGuesser(on_cuda=torch.cuda.is_available())
    dan_guesser.train(dataset.training_data())
    dan_guesser.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()
