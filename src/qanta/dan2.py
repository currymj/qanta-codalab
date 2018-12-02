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

MODEL_PATH = 'danmodel.pickle'
TORCH_MODEL_PATH = 'danmodel.pyt'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.999


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
                 n_hidden_units=256, n_additional_hidden_layers=0, nn_dropout=.1, word_drop=.1):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.word_drop = word_drop

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

        self.act = nn.ELU()
        self._softmax = nn.Softmax()
        self.dropout = nn.Dropout(nn_dropout)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden_units)
        classifier_layers = [self.linear1, self.batchnorm1, self.act, self.dropout]
        for _ in range(n_additional_hidden_layers):
            classifier_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            classifier_layers.append(nn.BatchNorm1d(n_hidden_units))
            classifier_layers.append(nn.ELU())
            classifier_layers.append(nn.Dropout(nn_dropout))
        classifier_layers.append(self.linear2)

        #### modify the init function, you need to add necessary layer definition here
        #### note that linear1, linear2 are used for mlp layer
        #self.hidden = nn.Linear(n_hidden_units, n_hidden_units)
        #self.classifier = nn.Sequential(self.linear1, self.batchnorm1, self.act, self.dropout, self.linear2)
        self.classifier = nn.Sequential(*classifier_layers)


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

        if self.training:
            word_drop_mask = (torch.rand(text_embed.shape[0], text_embed.shape[1], 1, device=text_embed.device) < 0.3)
            text_embed.masked_fill_(word_drop_mask, 0.0)
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

    def question_preprocessing(self, questions: List[str]) -> List[List[str]]:
        questions = preprocess.cleaning(questions, False)
        questions = [q.split() for q in questions]
        return questions

    def postprocess_answer(self, ans):
        "Replaces spaces with _ in the answers. Ideally, shouldn't be necessary."
        return '_'.join(ans.split())

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        self.dan_model.eval()
        questions_split = self.question_preprocessing(questions)
        input_questions = []
        for q in questions_split:
            input_questions.append(self.vectorize_without_labels(q))

        input_batch = self.batchify_without_labels(input_questions)
        question_text = input_batch['text']
        question_len = input_batch['len']
        logits = self.dan_model.forward(question_text, question_len, is_prob=True).detach()
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
    print("No training from within docker model. Train elsewhere and copy to the data directory.")


@cli.command()
def test_guess():
    dan_guesser = DanGuesser.load()
    print(dan_guesser.guess(['This is a test question for ten points.'], 1))
    print(dan_guesser.guess(['Here we have a first question for ten points.', 'And another question for ten points.'], 1))
    print(dan_guesser.guess(['Here we have a first question for ten points.', 'And another question for ten points.'], 2))

@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()
