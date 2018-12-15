"""
How to use:
- The class of interest here is Corpus. Familiarize yourself with its fields;
  they contain much of value.
- To create the corpus from .csv files, you'll merely need to write something
  like the following in a Jupyter notebook:

  > import corpus
  > qb_corpus = corpus.create_corpus(data_dir, batch_size)

  where `data_dir` is the directory containing the Quizbowl dataset .csv files,
  which are expected to be named "quizbowl_{train, dev, test}_final.csv".


- To use the Corpus object for training, merely access its PyTorch Dataloaders
  and use them as you would any other Dataloader. They're already set to
  randomly shuffle the dataset after every epoch:
  
  > train_loader = corpus.train_loader
  > for (batch_sents, batch_labels) in train_loader:
  >    ...

  There are also (naturally) `corpus.dev_loader` and `corpus.test_loader`
  fields.


- To save (pickle) the corpus, use the following:

  > corpus.save(qb_corpus)

  where qb_corpus is the Corpus object you'd like to pickle.
  Note: use this functionality at your own risk; it leads to
        memory errors on my machine.


NB:
- The Corpus object has as a field `glove_embeddings`. They have been pre-loaded
  for you. You may wish to use this to initialize the embedding layer of your
  model.

- Again, everything you need to train your model has been preprocessed and saved
  as fields in the Corpus object. You should familiarize yourself with them.
"""

# torch
import torch
from torch.utils.data import Dataset, RandomSampler, BatchSampler, SequentialSampler, DataLoader

# torchtext
import torchtext.vocab

# data science (yay!)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# numpy
import numpy as np

# tqdm
from tqdm import tqdm_notebook as tqdm

# nltk
import nltk
import nltk.tokenize
nltk.download('punkt')

# python
import os
import string
import pickle
import random
from collections import defaultdict

DATA_DIR = 'data'
CORPUS_DIR = DATA_DIR # os.path.join(DATA_DIR, 'corpus')

MIN_QUESTION_LENGTH = 5
MAX_SEQUENCE_LENGTH = 30

"""
EMBEDDINGS
"""

def load_embeddings():
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    print('Loaded {} words'.format(len(glove.itos)))
    
def get_embedding(word):
    return glove.vectors[glove.stoi[word]]

load_embeddings()

"""
UTILITIES
"""

def get_dataloader(sents, labels, cats, batch_size, max_sequence_length, vocab):
    dataset = SentenceDataset(sents, labels, cats, max_sequence_length, vocab)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)
        
    return dataloader

# create a table for stripping punctuation
table = str.maketrans('', '', string.punctuation)

"""
SAVE/LOAD
"""

def save(corpus):
    fn = '{}_bs{}_ml{}'.format(corpus.name, corpus.batch_size, corpus.max_sequence_length)
    fn = os.path.join(CORPUS_DIR, fn)
    with open(fn, 'wb') as f:
        pickle.dump(corpus, f)

def load(corpus_name):
    with open(os.path.join(CORPUS_DIR, corpus_name), 'rb') as f:
        corpus = pickle.load(f)
    
    return corpus

"""
CLASSES
"""

class Vocab:
    def __init__(self, vocab, i2w, w2i, counts):
        self.vocab = vocab
        self.i2w = i2w
        self.w2i = w2i
        self.len_ = len(i2w)
        self.counts = counts
        self._get_embeddings()
        
    def __len__(self):
        return self.len_
        
    def __getitem__(self, key):
        if type(key) is str:
            try:
                return self.w2i[key]
            except:
                return self.w2i['<unk>']
        else:
            try:
                key = int(key)
                return self.i2w[key]
            except:
                return '<unk>'
#         else:
#             raise Exception(("Invalid key '{}' -- neither int nor str;" +\
#                              "instead, type is {}".format(key, type(key)))
        
    def vocab(self):
        return self.vocab
    
    def i2w(self):
        return self.i2w
    
    def w2i(self):
        return self.w2i
    
    def _get_embeddings(self):
        ordered_vocab = [self[i] for i in range(len(self))]

        self.glove_embeddings = []
        for word in ordered_vocab:
            try:
                self.glove_embeddings.append(get_embedding(word).unsqueeze(0))
            except:
                self.glove_embeddings.append(torch.randn((1, 300)))

        self.glove_embeddings = torch.cat(self.glove_embeddings)

class Corpus:
    def __init__(self, name, vocab, cats, curriculum_len,
                 train_q_currs, train_a_currs,
                 train_sents, train_labels_v, train_labels, train_cats,
                 dev_sents, dev_labels_v, dev_labels, dev_cats,
                 test_sents, test_labels_v, test_labels, test_cats,
                 batch_size=32,
                 max_sequence_length=25,
                 classes=None,
                 silent=False
                ):
        # save properties
        self.name = name
        self.vocab = vocab
        self.cats = cats
        self.classes = classes
        self.curriculum_len = curriculum_len
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        
        self.train_q_currs = train_q_currs
        self.train_a_currs = train_a_currs
        self.train_sents = train_sents
        self.dev_sents = dev_sents
        self.test_sents = test_sents
        self.train_labels_v = train_labels_v
        self.dev_labels_v = dev_labels_v
        self.test_labels_v = test_labels_v
        self.train_labels = train_labels
        self.dev_labels = dev_labels
        self.test_labels = test_labels
        self.train_cats = train_cats
        self.dev_cats = dev_cats
        self.test_cats = test_cats
        
        # compute some corpus statistics
        self.num_types = len(self.vocab)
        self.num_tokens = len(self.train_sents) + len(self.dev_sents) + len(self.test_sents)
        self.num_classes = len(classes) if classes else 0
        self.num_cats = len(self.cats)
        
        # create dataloaders
        self.train_loader = get_dataloader(train_sents,
                                           train_labels_v,
                                           train_cats,
                                           self.batch_size,
                                           max_sequence_length,
                                           self.vocab)
        self.dev_loader = get_dataloader(dev_sents,
                                         dev_labels_v,
                                         dev_cats,
                                         self.batch_size,
                                         max_sequence_length,
                                         self.vocab)
        self.test_loader = get_dataloader(test_sents,
                                          test_labels_v,
                                          test_cats,
                                          self.batch_size,
                                          max_sequence_length,
                                          self.vocab)
        
        if train_q_currs and train_a_currs:
            self.train_loaders = [get_dataloader(train_sents,
                                                train_labels,
                                                self.batch_size,
                                                max_sequence_length,
                                                self.vocab) for
                                  (train_sents, train_labels) in zip(train_q_currs, train_a_currs)]
        else:
            self.train_loaders = None
        
        # report some corpus statistics
        if not silent:
            print("Num word tokens: {}".format(self.num_tokens))
            print("Num word types: {}".format(self.num_types))
            print("Num question categories: {}".format(self.num_cats))
            print("Num classes: {}".format(self.num_classes))
            print("Num curricula: {}".format(self.curriculum_len))
            print("Train size: {}".format(len(self.train_sents)))
            print("Dev size: {}".format(len(self.dev_sents)))
            print("Test size: {}".format(len(self.test_sents)))
            if self.train_q_currs:
                print("Curriculum train size: {}".format(sum([len(sents) for sents in self.train_q_currs])))
            
class SentenceDataset(Dataset):
    def __init__(self, sents, labels, cats, max_sequence_length, vocab):
        super(Dataset, self).__init__()
        self.sents = sents
        self.labels = labels
        self.cats = cats
        self.vocab = vocab
        self.len_ = len(self.sents)
        
    def __len__(self):
        return self.len_
    
    def __getitem__(self, i):
        sent, length = numericalize_sent(self.sents[i], self.vocab, MAX_SEQUENCE_LENGTH)
        label = self.labels[i]
        cat = self.cats[i]
        
        return (sent, length, cat), label
    
"""
CORPUS CREATION
"""

def create_corpus(data_dir, batch_size, max_sequence_length=None,
                  augment_train=True, augment_mode='random',
                  augment_dev=False, min_occurrences=4,
                  curriculum_len=None, pranav=True):
    
    if pranav:
        train_file = os.path.join(data_dir, "quizbowl_train_final.csv")
        print("Train Preprocessing")
        print("-------------------")
        train_sents, train_labels, train_cats = process_qb_csv(train_file,
                                                           do_augment=augment_train,
                                                           augment_mode=augment_mode)
        
        tuples = [(s, l, c) for (s, l, c) in zip(train_sents, train_labels, train_cats) if len(s) >= MIN_QUESTION_LENGTH]
        train_sents, train_labels, train_cats = zip(*tuples)
        train_sents, train_labels, train_cats = list(train_sents), list(train_labels), list(train_cats)
        
        _, _, dev_sents, dev_labels, c2i, i2c = pranav_craziness()
        
        # train_sents = [q for (q, l) in train_sents]
        dev_sents = [q for (q, l) in dev_sents]
        
        vocab = process_vocab(train_sents + dev_sents)
        classes = Vocab(vocab=None, i2w=i2c, w2i=c2i, counts=None)

        cats = []
        train_cats = [0 for l in train_labels]
        dev_cats = [0 for l in dev_labels]

        # numericalize train labels
        train_labels_v = numericalize_labels(train_labels, classes=classes)
        
        # to torch
        print("Moving labels to torch ...")
        train_labels_v = torch.tensor(train_labels_v)
        dev_labels_v = torch.tensor(dev_labels)
        
        test_sents = []
        test_labels_v = []
        test_labels = []
        test_cats = []
        
        if not max_sequence_length:
            max_sequence_length = max([len(q) for q in train_sents + dev_sents])
            print("Calculated maximum sequence length: {}".format(max_sequence_length))
        
        train_q_currs = None
        train_a_currs = None
        
        print("Creating corpus object ...")
        # consolidate into dataset
        corpus = Corpus('quizbowl', vocab, cats, curriculum_len,
                        train_q_currs, train_a_currs,
                        train_sents, train_labels_v, train_labels, train_cats,
                        dev_sents, dev_labels_v, dev_labels, dev_cats,
                        test_sents, test_labels_v, test_labels, test_cats,
                        batch_size, max_sequence_length,
                        classes=classes)

        print("\nFinished!")
        return corpus
        
    
    # file paths
    # qb_dir = os.path.join(data_dir, 'quizbowl')
    train_file = os.path.join(data_dir, "quizbowl_train_final.csv")
    dev_file = os.path.join(data_dir, "quizbowl_dev_final.csv")
    test_file = os.path.join(data_dir, "quizbowl_test_final.csv")
    
    train_q_currs, train_a_currs = None, None

    # process
    print("Train Preprocessing")
    print("-------------------")
    train_sents, train_labels, train_cats = process_qb_csv(train_file,
                                                           do_augment=augment_train,
                                                           augment_mode=augment_mode)
    print("\n")
    
    print("Dev Preprocessing")
    print("-------------------")
    dev_sents, dev_labels, dev_cats = process_qb_csv(dev_file,
                                                     do_augment=augment_dev,
                                                     augment_mode=augment_mode)
    print("\n")
    
    print("Test Preprocessing")
    print("-------------------")
    test_sents, test_labels, test_cats = process_qb_csv(test_file, do_augment=False)
    print("\n")
    
    if not max_sequence_length:
        max_sequence_length = max([len(q) for q in train_sents + dev_sents + test_sents])
        print("Calculated maximum sequence length: {}".format(max_sequence_length))
    
    # build vocab and labels
    print("Building vocab ...")
    vocab = process_vocab(train_sents + dev_sents + test_sents, min_occurrences=min_occurrences)
    print("Building label set ...")
    classes = process_labels(train_labels + dev_labels + test_labels)
    print("Building category set ...")
    cats = process_labels(train_cats + dev_cats + test_cats)
    
    tuples = [(s, l, c) for (s, l, c) in zip(train_sents, train_labels, train_cats) if len(s) >= MIN_QUESTION_LENGTH]
    train_sents, train_labels, train_cats = zip(*tuples)
    
    print("Numericalizing labels ...")
    train_labels_v = numericalize_labels(train_labels, classes=classes)
    dev_labels_v = numericalize_labels(dev_labels, classes=classes)
    test_labels_v = numericalize_labels(test_labels, classes=classes)
    
    print("Numericalizing categories ...")
    train_cats = numericalize_labels(train_cats, classes=cats)
    dev_cats = numericalize_labels(dev_cats, classes=cats)
    test_cats = numericalize_labels(test_cats, classes=cats)
    
    # to torch
    print("Moving labels to torch ...")
    train_labels_v = torch.tensor(train_labels_v)
    dev_labels_v = torch.tensor(dev_labels_v)
    test_labels_v = torch.tensor(test_labels_v)
    
    train_cats = torch.tensor(train_cats)
    dev_cats = torch.tensor(dev_cats)
    test_cats = torch.tensor(test_cats)
    
    # create curricula
    if curriculum_len:
        print("Train curriculum preprocessing")
        print("-------------------")
        train_q_currs, train_a_currs = process_qb_csv_curr(train_file, curriculum_len)
        train_a_currs = [torch.tensor(train_labels) for train_labels in train_a_currs]
        train_a_currs = numericalize_label_currs(train_a_currs, classes=classes)
        
    
    # for i, answer in enumerate(train_labels):
    #     if len(answer) == 1:
    #         print("problem with index {}".format(i))
    #         print("\toriginal answer: {}".format(answer))
    #         print("\toriginal question: {}".format(train_sents[i]))
    
    print("Creating corpus object ...")
    # consolidate into dataset
    corpus = Corpus('quizbowl', vocab, cats, curriculum_len,
                    train_q_currs, train_a_currs,
                    train_sents, train_labels_v, train_labels, train_cats,
                    dev_sents, dev_labels_v, dev_labels, dev_cats,
                    test_sents, test_labels_v, test_labels, test_cats,
                    batch_size, max_sequence_length,
                    classes=classes)
    
    print("\nFinished!")
    return corpus

def pranav_craziness():
    import preprocess
    
    train_exs_orig = preprocess.read_csv('data/quizbowl_train_final.csv')
    dev_exs_orig = preprocess.read_csv('data/quizbowl_dev_final.csv')
    
    class_to_i, i_to_class = preprocess.class_labels(train_exs_orig + dev_exs_orig)
    train_exs, dev_exs= [], []
    
    for x in train_exs_orig:
        train_exs.append((x[0], class_to_i[x[1]]))
    for x in dev_exs_orig:
        dev_exs.append((x[0], class_to_i[x[1]]))
    y_train, y_dev = [], []
    for x in train_exs:
        y_train.append(x[1])
    y_train = np.array(y_train)
    for x in dev_exs:
        y_dev.append(x[1])
    y_dev = np.array(y_dev)
    
    return train_exs, y_train, dev_exs, y_dev, class_to_i, i_to_class

"""
QUESTION-TEXT PROCESSING UTILITIES
"""

def remove_infrequent_words(sents, vocab, min_occurrences=3):
    if not min_occurrences:
        return sents
    
    cleaned_sents = []
    for sent in tqdm(sents):
        cleaned_sent = []
        for word in sent:
            if min_occurrences and vocab.counts[word] < min_occurrences:
                cleaned_sent.append('<unk>')
            else:
                cleaned_sent.append(word)
        cleaned_sents.append(cleaned_sent)
                
    return cleaned_sents

def clean(token):
    if type(token) is float:
        print(True)
    # clean up the token
    token = token.lower()
    token = token.translate(table)
    
    return token

def process_qb_csv(filename, do_augment=True, augment_mode='random'):
    # read in the questions
    print("Reading in questions ...")
    df = pd.read_csv(filename)
    questions = df['Text'].values.tolist()
    answers = df['Answer'].values.tolist()
    categories = df['Category'].values.tolist()
    
    # process the questions
    print("Processing questions ...")
    questions, answers, categories = process_questions(questions, answers, categories,
                                                       do_augment, augment_mode=augment_mode)
    
    return questions, answers, categories

def process_qb_csv_curr(filename, curriculum_len=3):
    # read in the questions
    print("Reading in questions ...")
    df = pd.read_csv(filename)
    questions = df['Text'].values.tolist()
    
    # get the answers
    print("Processing answers ...")
    answers = []
    for answer in tqdm(df['Answer'].values.tolist()):
        if type(answer) is str:
            answers.append(clean(answer))
    
    # process the questions
    print("Processing questions ...")
    q_currs, a_currs = process_questions_curr(questions, answers,
                                              curriculum_len=curriculum_len)
    
    return q_currs, a_currs

def augment(question, answer, category):
    sent_parts = nltk.tokenize.sent_tokenize(question)
    sent_parts = [part.strip() for part in sent_parts]
    # clause_parts = [clause.strip() for part in sent_parts for clause in part.split(';')]
    
    if len(sent_parts) == 1:
        # print("category: {}".format(category))
        return sent_parts, [answer], [category]
    
    acc = [sent_parts[0]]
    for part in sent_parts[1:]:
        acc.append(acc[-1] + ' ' + part)
        
    questions = list(set(acc))
    
    # print("category: {}".format(category))

    return questions, [answer] * len(questions), [category] * len(questions)

def augment_random(question, answer, category):
    sent_parts = nltk.tokenize.sent_tokenize(question)
    sent_parts = [part.strip() for part in sent_parts]
    clause_parts = [clause.strip() for part in sent_parts for clause in part.split(';')]
    num_clauses = len(clause_parts)

    if num_clauses == 1:
        questions = clause_parts
        questions = [q for q in questions if len(q) > 0]
        
        if len(questions) > 0:
            return questions, [answer] * len(questions), [category] * len(questions)
        else:
            return [], [], []
    elif num_clauses >= 2:
        questions = []
        for i in range(5):
            start = random.randint(0, num_clauses - 1)
            end = random.randint(start + 1, num_clauses)
            question = ' '.join(clause_parts[start:end])
            questions.append(question)

        # include semi-colon separated clauses individuall (as factoids)
        questions += clause_parts
        # remove duplicate questions
        questions = list(set(questions))
        # remove stray empty questions
        questions = [q for q in questions if len(q) > 0]
        
        return questions, [answer] * len(questions), [category] * len(questions)

def augment_curr(question, answer, curriculum_len=3):
    sent_parts = nltk.tokenize.sent_tokenize(question)
    clause_parts = [clause.strip() for part in sent_parts for clause in part.split(';')]
    
    if len(clause_parts) == 1:
        clause_parts += [] * (curriculum_len - 1)
        answer += [] * (curriculum_len - 1)
        return clause_parts, answer
    
    acc = [clause_parts[0]]
    for part in clause_parts[1:]:
        acc.append(acc[-1] + ' ' + part)
        
    questions = list(set(acc + sent_parts))

    return questions, [answer] * len(questions)

def process_questions(questions, answers, categories, do_augment=True,
                      augment_mode='sequential'):
    # split the questions into sentences and clauses, if desired
    if do_augment:
        augment_mode == None
        if augment_mode == 'sequential':
            print("Using sequential augmentation ...")
            augment_fn = augment
        elif augment_mode == 'random':
            print("Using random augmentation ...")
            augment_fn = augment_random
        else:
            raise Error("Invalid augment mode '{}'".format(augment_mode))
        
        ts = zip(questions, answers, categories)
        print("Split the questions into sentences and clauses and aggregate into subsets ...")
        questions = []
        answers = []
        categories = []
        for i, t in enumerate(tqdm(list(ts))):
            # get the augmented question/answer pair (the answer for each
            # question may have to be duplicated a number of times as the question
            # is broken apart)
            question_parts, answer_parts, category_parts = augment_fn(t[0], t[1], t[2])
            # accumulate the results
            if len(question_parts) > 0:
                questions += question_parts
                answers += answer_parts
                categories += category_parts
    
    # tokenize each sentence part
    questions = tokenize_sents(questions)
    
    # clean the questions up
    questions = clean_sents(questions)
    
    # strip non-word characters
    questions = strip_non_words(questions)
    
    return questions, answers, categories

def process_questions_curr(questions, answers, curriculum_len=3):    
    # split the questions into sentences and clauses, if desired
    pairs = zip(questions, answers)
    print("Split the questions into sentences and clauses and aggregate into subsets ...")
    questions = []
    answers = []
    for pair in tqdm(list(pairs)):
        # get the augmented question/answer pair (the answer for each
        # question may have to be duplicated a number of times as the question
        # is broken apart)
        question_parts, answer_parts = augment(pair[0], pair[1],
                                               curriculum_len=curriculum_len)
        # accumulate the results
        questions += question_parts
        answers += answer_parts
            
    question_curricula = [list(itertools.chain.from_iterable(curr)) for curr in list(zip(*questions))]
    answer_curricula = [list(itertools.chain.from_iterable(curr)) for curr in list(zip(*answers))]
    
    # tokenize each sentence part
    question_curricula = [tokenize_sents(qs) for qs in question_curricula]
    
    # clean the questions up
    question_curricula = [clean_sents(qs) for qs in question_curricula]
    
    # strip non-word characters
    question_curricula = [strip_non_word(qs) for qs in question_curricula]
    
    return question_curricula, answer_curricula

def tokenize_sents(questions):
    print("Tokenizing each sentence part ...")
    temp = []
    for question in tqdm(questions):
        temp.append([word for word in question.split()])
    questions = temp
    
    return questions
    
def clean_sents(questions):
    print("Cleaning up each sentence part ...")
    temp = []
    for question in tqdm(questions):
        temp.append([clean(word) for word in question])
    questions = temp
    
    return questions
    
def strip_non_words(questions):
    print("Stripping non-word characters ...")
    temp = []
    for question in tqdm(questions):
        temp.append([word for word in question if word.isalpha()])
    questions = temp
    
    return questions

def numericalize_sent(sent, vocab, max_sequence_length=None, pad=True):
    sent = [vocab[word] for word in sent]
    length = len(sent)
    sent = sent[-max_sequence_length:]
    length = min(length, len(sent))
    
    # pad if desired
    if pad:
        template = [vocab['<pad>']] * max_sequence_length
        sent = sent + template[len(sent):]
    
    return torch.tensor(sent), length

def numericalize_labels(labels, classes):
    temp = []
    for label in tqdm(labels):
        temp.append(classes[label])
    
    return np.array(temp)

def numericalize_label_currs(currs, classes):
    temp_currs = []
    for curr in currs:
        temp_curr = []
        for label in tqdm(labels):
            temp_curr.append(classes[label])
        temp_currs.append(temp_curr)
    
    return temp_currs

"""
VOCABULARY PREPROCESSING
"""

def process_vocab(sents, min_occurrences=3):
    # build the vocabulary, and i2w/w2i dictionaries
    reserved = ['<pad>', '<unk>']
    print("Collecting lowercase wordset:")
    
    counts = defaultdict(int)
    words = []
    for sent in tqdm(sents):
        for word in sent:
            words.append(word.lower())
        counts[word] += 1
    
    words = list(set(words))
    if min_occurrences:
        words = [word for word in words if counts[word] >= min_occurrences]
    
    print("Building word list (i2w) ...")
    words += reserved
    i2w = list(words)
    print("Building word2index dictionary (w2i) ...")
    w2i = defaultdict(int, {w: i for i, w in tqdm(enumerate(i2w))})
    
    return Vocab(vocab=words, i2w=i2w, w2i=w2i, counts=counts)

"""
LABEL PREPROCESSING
"""

def process_labels(classes):
    classes = set(classes)
    i2c = list(classes)
    c2i = {c: i for i, c in tqdm(enumerate(i2c))}
    
    return Vocab(vocab=classes, i2w=i2c, w2i=c2i, counts=None)