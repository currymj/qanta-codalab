import pandas as pd
import re
import string
from typing import List
from nltk import word_tokenize
from nltk.corpus import stopwords


def read_csv(filename, remove_stopwords=False):
    ## Reading data
    df = pd.read_csv(filename)
    questions = df['Text'].values.tolist()
    answers = df['Answer'].values.tolist()

    ## cleaning data (trivial preprocessing and removing ftp)
    questions = cleaning(questions, remove_stopwords)
    questions = [q.split() for q in questions]
    data = list(zip(questions, answers))
    return data


def cleaning(q, remove_stopwords):
    ftp_patterns = {
        '\n',
        ', for 10 points,',
        ', for ten points,',
        '--for 10 points--',
        'for 10 points, ',
        'for 10 points--',
        'for ten points, ',
        'for 10 points ',
        'for ten points ',
        ', ftp,'
        'ftp,',
        'ftp',
        'FTP'
    }

    patterns = ftp_patterns | set(string.punctuation)
    regex_pattern = '|'.join([re.escape(p) for p in patterns])
    regex_pattern += r'|\[.*?\]|\(.*?\)'

    q = [re.sub(regex_pattern,' ', element.strip().lower()) for element in q]
    q = [re.sub(r'[^\x00-\x7F]+',' ', element) for element in q]
    q = [' '.join(element.split()) for element in q]
    if remove_stopwords:
        stop_words = stopwords.words('english')
        for i in range(len(q)):
            words = q[i].split()
            q[i] = ' '.join([w for w in words if not w in stop_words])
    
    return q
def load_words(exs):
    """
    vocabuary building
    Keyword arguments:
    exs: list of input questions-type pairs
    """
    words = set()
    UNK = '<unk>'
    PAD = '<pad>'
    word2ind = {PAD: 0, UNK: 1}
    ind2word = {0: PAD, 1: UNK}
    for q_text, _ in exs:
        #q_text = word_tokenize(q_text)
        #q_text = q_text.split()
        for w in q_text:
            words.add(w)
    words = sorted(words)
    for w in words:
        idx = len(word2ind)
        word2ind[w] = idx
        ind2word[idx] = w
    words = [PAD, UNK] + words
    return words, word2ind, ind2word

def class_labels(data):
    class_to_i = {}
    i_to_class = {}
    i = 0
    for _, ans in data:
        if ans not in class_to_i.keys():
            class_to_i[ans] = i
            i_to_class[i] = ans
            i+=1
    return class_to_i, i_to_class

if __name__ == "__main__":
    ## File paths
    train_file = "quizbowl_train_final.csv"
    val_file = "quizbowl_dev_final.csv"
    test_file = "quizbowl_test_final.csv"

    ## File processing and cleaning
    train_data = read_csv(train_file)
    #print(train_data[1:5])

    #words, word2ind, ind2word = load_words(train_data)


    # print(len(words))
    # print(words[100000:100025])

    # print(word2ind)


    dev_data = read_csv(val_file)

    class_to_i, i_to_class = class_labels(train_data + dev_data)
    #print(class_to_i)

    test_data = read_csv(test_file)
    print('Done Reading files')
