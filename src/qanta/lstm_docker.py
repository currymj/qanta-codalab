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

from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F



MODEL_PATH = 'lstm_model.pickle'
TORCH_MODEL_PATH = 'lstm_model_1.pyt'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    print('guessing')
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    print('batch guessing')
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs

class QAmodel(nn.Module):
    def __init__(self, n_classes,vocab, model1,word_embedding_dim=300, hidden_dim=100, train_vocab_embeddings=None):
        super(QAmodel, self).__init__()
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, word_embedding_dim, padding_idx=0)

        if train_vocab_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(train_vocab_embeddings))
        else:
            pretrained_weights = np.zeros((self.vocab_size, 300))

            for i, word in enumerate(vocab):
                try:
                    pretrained_weights[i] = model1[word]
                except:
                    pretrained_weights[i] = model1['unk']
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))

        self.hidden_size = hidden_dim
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, num_layers=1,
            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.dense1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.output = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, questions, lens):
        bsz, max_len = questions.size()
        embeds = self.dropout(self.embedding(questions))

        lens, indices = torch.sort(lens, 0, True)
        _, (enc_hids, _) = self.lstm(pack(embeds[indices], lens.tolist(), batch_first=True))
        enc_hids = torch.cat( (enc_hids[0], enc_hids[1]), 1)
        _, _indices = torch.sort(indices, 0)
        enc_hids = enc_hids[_indices]
        dense_output = self.dense1(enc_hids)
        output = self.output(dense_output)
        return F.log_softmax(output)

class LstmGuesser:
    def __init__(self, device='cpu'):
        self.lstm_model = None
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

    def question_preprocessing(self, questions: List[str]) -> List[List[str]]:
        questions = preprocess.cleaning(questions, False)
        questions = [q.split() for q in questions]
        return questions

    def postprocess_answer(self, ans):
        "Replaces spaces with _ in the answers. Ideally, shouldn't be necessary."
        return '_'.join(ans.split())

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        self.lstm_model.eval()
        questions_split = self.question_preprocessing(questions)
        input_questions = []
        for q in questions_split:
            input_questions.append(self.vectorize_without_labels(q))

        input_batch = self.batchify_without_labels(input_questions)
        question_text = input_batch['text']
        question_len = input_batch['len']
        logits = self.lstm_model.forward(question_text, question_len).detach()
        top_n, top_i = logits.topk(max_n_guesses)
        answer_indices = top_i.numpy()
        answer_scores = top_n.numpy()
        answer_score_pair_lists = []
        for i in range(len(answer_indices)):
            q_top_answers = [self.postprocess_answer(self.i_to_class[ans_ind]) for ans_ind in answer_indices[i]]
            q_top_scores = [score for score in answer_scores[i]]
            answer_score_pair_lists.append( list(zip(q_top_answers, q_top_scores)) )

        return answer_score_pair_lists


    def batchify_without_labels(self, batch):
        """
        Gather a batch of individual examples into one batch, 
        which includes the question text, question length and labels 
        Keyword arguments:
        batch: list of outputs from vectorize function
        """

        question_len = list()
        for ex in batch:
            question_len.append(len(ex))
        x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
        for i in range(len(question_len)):
            question_text = batch[i]
            vec = torch.LongTensor(question_text)
            x1[i, :len(question_text)].copy_(vec)
        q_batch = {'text': x1, 'len': torch.FloatTensor(question_len)}
        return q_batch

    def vectorize_with_labels(self, ex):
        """
        vectorize a single example based on the word2ind dict. 
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        question_text, question_label = ex
        vec_text = [0] * len(question_text)

        for i in range(len(question_text)):
            try:
                vec_text[i] = self.word2ind[question_text[i]]
            except:
                vec_text[i] = self.word2ind['<unk>']

        return vec_text, question_label

    def vectorize_without_labels(self, ex):
        """
        vectorize a single example based on the word2ind dict. 
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        question_text = ex
        vec_text = [0] * len(question_text)

        for i in range(len(question_text)):
            try:
                vec_text[i] = self.word2ind[question_text[i]]
            except:
                vec_text[i] = self.word2ind['<unk>']

        return vec_text

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = LstmGuesser()
            guesser.i_to_class = params['i_to_class']
            guesser.class_to_i = params['class_to_i']
            guesser.voc = params['voc']
            guesser.ind2word = params['ind2word']
            guesser.word2ind = params['word2ind']
            num_classes = params['num_classes']
            guesser.glove_model = params['glove_model']
            guesser.lstm_model = QAmodel(num_classes, guesser.voc, guesser.glove_model)
            guesser.lstm_model.load_state_dict(torch.load(
                TORCH_MODEL_PATH))
            guesser.lstm_model.eval()
        return guesser


def create_app(enable_batch=True):
    dan_guesser = LstmGuesser.load()
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
    Start web server wrapping lstm model
    """
    print('lstm app running')
    app = create_app(enable_batch=False)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the DAN model, requires downloaded data and saves to models/
    """
    print("No training from within docker model. Train elsewhere and copy to the data directory.")


@cli.command()
def test_guess():
    dan_guesser = LstmGuesser.load()
    print(dan_guesser.guess(['This is a test question for ten points.'], 1))
    print(dan_guesser.guess(['Here we have a first question for ten points.', 'And another question for ten points.'], 1))
    print(dan_guesser.guess(['Here we have a first question for ten points.', 'And another question for ten points.'], 2))
    print(guess_and_buzz(dan_guesser, 'This is a test question for ten points.'))
    print(batch_guess_and_buzz(dan_guesser, ['This is a test question for ten points.']))

@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()