from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path
import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from qanta import util
from qanta.dataset import QuizBowlDataset
from qanta import preprocess
import torch
from torch import nn
import nltk

MODEL_PATH = 'danmodel.pickle'
TORCH_MODEL_PATH = 'danmodel.pyt'
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

class DanEncoder(nn.Module):
    # from QANTA laboratory
    def __init__(self, embedding_dim, n_hidden_layers, n_hidden_units, dropout_prob):
        super(DanEncoder, self).__init__()
        encoder_layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                input_dim = embedding_dim
            else:
                input_dim = n_hidden_units

            encoder_layers.extend([
                nn.Linear(input_dim, n_hidden_units),
                nn.BatchNorm1d(n_hidden_units),
                nn.ELU(),
                nn.Dropout(dropout_prob),
            ])
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x_array):
        return self.encoder(x_array)

class DanModel(nn.Module):
    def __init__(self, n_classes, n_hidden_units, embedding_dim, vocab_size, pad_idx, nn_dropout):
        super(DanModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_hidden_units = n_hidden_units
        self.encoder = DanEncoder(embedding_dim, 2, n_hidden_units ,0.1)
        self.n_classes = n_classes
        self.nn_dropout = nn_dropout
        self.classifier = nn.Sequential(
                nn.Linear(self.n_hidden_units, n_classes),
                nn.BatchNorm1d(self.n_classes),
                nn.Dropout(self.nn_dropout)
        )
        self.dropout = nn.Dropout(self.nn_dropout)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        #self.unigram_embeddings.weight.data = 
    def _pool(self, embed):
        emb_max, _ = torch.max(embed, 1)
        return emb_max

    def forward(self, input_words):
        embed = self.embedding(input_words)
        embed = self._pool(embed)
        embed = self.dropout(embed)
        encoded = self.encoder(embed)
        return self.classifier(encoded)




class DanGuesser:
    def __init__(self):
        self.dan_model = None
        self.i_to_ans = None
        self.ans_to_i = None
        self.vocab_list = None
        self.i_to_word = None
        self.word_to_i = None

    def train(self, training_data) -> None:
        hidden_layer_size = 256
        questions = training_data[0]
        answers = training_data[1]
        words, self.word_to_i, self.i_to_word = preprocess.load_words(questions)
        self.i_to_ans = {i: ans for i, ans in enumerate(answers)}
        self.ans_to_i = dict((v,k) for k,v in self.i_to_ans.items())
        self.dan_model = DanModel(len(self.ans_to_i),hidden_layer_size, hidden_layer_size, len(words), 0, 0.1)

        print('Questions loaded, now iterating...')
        for q in questions:
            batched = self.questions_to_batch([q])
            print(batched)
            torch_tensor = self.make_padded_tensor(batched)
            self.dan_model(torch_tensor)

    def make_padded_tensor(self, batch_inds):
        lengths = [len(q) for q in batch_inds]
        pad_len = max(lengths)
        torch_tensor = torch.zeros(len(batch_inds), pad_len, dtype=torch.int64)
        for i in range(len(batch_inds)):
            for j in range(len(batch_inds[i])):
                torch_tensor[i, j] = batch_inds[i][j]
        return torch_tensor

    def questions_to_batch(self, question_strings):
        tokenized = [preprocess.word_to_tokens(q, self.word_to_i) for q in question_strings]
        return tokenized


    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({'i_to_ans': self.i_to_ans, 'ans_to_i': self.ans_to_i},
                    f)
        torch.save(self.dan_model.state_dict(), TORCH_MODEL_PATH)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        #idx_qs = self.questions_to_batch(questions)
        #prediction_scores = self.dan_model(idx_qs)

        guesses = []
        for i in range(len(questions)):
            guesses.append([('beer', 1.0), ('coffee', 0.5)])
        return guesses

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = DanGuesser()
            guesser.i_to_ans = params['i_to_ans']
            guesser.ans_to_i = params['ans_to_i']
            guesser.dan_model = DanModel(100,256, 256, len(guesser.ans_to_i), 0, 0.1)
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
    print('Downloading punkt...')
    nltk.download('punkt')
    print('training DAN...')
    dan_guesser = DanGuesser()
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
