# torch
import torch

# numpy
import numpy as np
import corpus as corpus_p

# data science! yay
import matplotlib.pyplot as plt

# python
import math
from collections import defaultdict

def get_answers(questions, model, corpus, k):
    model.eval()
    
    answers = [get_answer(question, model, corpus, k) for question in questions]
    answers = torch.tensor(answers).squeeze()
    
    return answers

def get_answer(question, model, corpus, k):    
    question, length = corpus_p.numericalize_sent(question,
                                                  corpus.vocab,
                                                  max_sequence_length=corpus.max_sequence_length)

    question = question.view(1, -1).cuda()
    length = torch.tensor([length]).cuda()

    log_ps = model.forward(question, lens=length, cats=None, training=False)

    answer_idx = torch.topk(log_ps, k=k, dim=-1)[1]

    return answer_idx

def exponential_smoothing(ys, beta=0.8, ub=math.inf, lb=-math.inf):
    """
    This is ugly, and I should have used a comprehension, but
    it'll get the job done. I made it a function because I suspect
    I may need it later.
    """
    smooth_ys = [ys[0]]
    for y in ys:
        if y > ub or y < lb:
            smooth_ys.append(smooth_ys[-1])
        else:
            smooth_ys.append(beta * smooth_ys[-1] + (1 - beta) * y)
    return smooth_ys[1:]

def plot(ELBO, NLL, KL, title, xlabel="Batches", ylabel="Measurements", hline=None):
    """
    Just a *slight* abstraction over pyplot to ease development a bit.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if hline:
        plt.axhline(y=hline, color='r', linestyle='-')
    
    plt.plot(list(range(len(ELBO))), ELBO, label="ELBO")
    plt.plot(list(range(len(NLL))), NLL, label="NLL Loss", c='blue')
    plt.plot(list(range(len(KL))), KL, label="KL Loss", c='red')
    plt.legend()
        
    plt.show()
    
tracker = defaultdict(list)
def plot_elbo(ELBO=tracker['ELBO'], title='ELBO', xlabel="Epochs", ylabel="ELBO", hline=None):
    """
    Just a *slight* abstraction over pyplot to ease development a bit.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(list(range(len(ELBO))), ELBO, label="ELBO")
    plt.legend()
        
    plt.show()
    
def plot_train_test(train, test, title, xlabel='Epochs', ylabel='ELBO'):
    """
    Just a *slight* abstraction over pyplot to ease development a bit.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(list(range(len(train))), train, label="Train")
    plt.plot(list(range(len(test))), test, label="Test")
    plt.legend()
        
    plt.show()
    
def print_logits(logits, vocab):
    dec = []
    for logp in logps:
        dec.append(vocab[torch.topk(logp, k=1)[1].item()])
        if dec[-1] == '<eos>': break
    
    print(' '.join(dec))
    
def print_idxs(idxs, vocab):
    dec = []
    if type(idxs) == torch.Tensor:
        idxs = idxs.squeeze()
    for i in idxs:
        dec.append(vocab[i])
        if dec[-1] == '<eos>': break
    
    print(' '.join(dec))
    
def interpolate(start, end, steps):
    start.squeeze()
    end.squeeze()

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T

def eval_by_gen(model, vocab, n=10):
    for i in range(n):
        print_idxs(model.generate(vocab)[0], vocab)