#!/usr/bin/env python
# coding: utf-8

#To analyze movie reviews and build a sentiment classification model
#Author: Akshay Mattoo

import pandas as pd
import os

train_data_pos = []
abs_path = "/Users/akshaymattoo/Desktop/movie_rating_pred/aclImdb/train/pos/"
for file in os.listdir(abs_path):
    curr_path = abs_path + file
    with open(curr_path, 'r') as fd:
        data = fd.read()
        data = data.lower()
        train_data_pos.append(data)
        
train_data_neg = []
abs_path = "/Users/akshaymattoo/Desktop/movie_rating_pred/aclImdb/train/neg/"
for file in os.listdir(abs_path):
    curr_path = abs_path + file
    with open(curr_path, 'r') as fd:
        data = fd.read()
        data = data.lower()
        train_data_neg.append(data)


from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer("[a-z1-9]+")
english_stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

wanted_stopwords = ["not", "wasn't", "won", "lose", "lost", "don't", "down", "any", "isn't", "yourself", "won't", "didn't", "should've", "should", "did", "doesn't", "our"]
for word in wanted_stopwords:
    if word in english_stopwords:
        english_stopwords.remove(word)

def cleanData(data):
    data = data.replace("<br /><br />"," ")
    
    #Tokenize the data
    tokens = tokenizer.tokenize(data)
    cleaned_tokens = [token for token in tokens if token not in english_stopwords]
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
    lem_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    return lem_tokens


cleaned_pos_data = [cleanData(data) for data in train_data_pos]
cleaned_neg_data = [cleanData(data) for data in train_data_neg]

vocab_size = 0;
pos_word_freq = {}
pos_word_count = 0
for review in cleaned_pos_data:
    for word in review:
        pos_word_count += 1
        pos_word_freq.setdefault(word, 1)
        if (pos_word_freq[word]==1):
            vocab_size += 1
        pos_word_freq[word] += 1
    
neg_word_freq = {}
neg_word_count = 0
for review in cleaned_neg_data:
    for word in review:
        neg_word_count += 1
        neg_word_freq.setdefault(word, 1)
        if (neg_word_freq[word]==1):
            vocab_size += 1
        neg_word_freq[word] += 1


total_count = 50000
pos_count = 25000
neg_count = 25000

prior_prob_pos = pos_count/total_count
prior_prob_neg = neg_count/total_count


test_data_pos = []
abs_path = "/Users/akshaymattoo/Desktop/movie_rating_pred/aclImdb/test/pos/"
for file in os.listdir(abs_path):
    curr_path = abs_path + file
    with open(curr_path, 'r') as fd:
        data = fd.read()
        data = data.lower()
        test_data_pos.append(data)

test_data_neg = []
abs_path = "/Users/akshaymattoo/Desktop/movie_rating_pred/aclImdb/test/neg/"
for file in os.listdir(abs_path):
    curr_path = abs_path + file
    with open(curr_path, 'r') as fd:
        data = fd.read()
        data = data.lower()
        test_data_neg.append(data)    
        
cleaned_test_pos = [cleanData(data) for data in test_data_pos]
cleaned_test_neg = [cleanData(data) for data in test_data_neg]


correct_pred_count = 0;

for review in cleaned_test_pos:
    pos_prob = 1
    neg_prob = 1
    
    for word in review:
        pos_likelihood = (1 / (pos_word_count + vocab_size))
        if word in pos_word_freq:
            pos_likelihood = ((pos_word_freq[word] + 1) / (pos_word_count + vocab_size))
        pos_prob *= pos_likelihood
        neg_likelihood = (1 / (neg_word_count + vocab_size))
        if word in neg_word_freq:
            neg_likelihood = ((neg_word_freq[word] + 1) / (neg_word_count + vocab_size))
        neg_prob *= neg_likelihood
    

    neg_prob *= prior_prob_neg
    pos_prob *= prior_prob_pos
     
    if (pos_prob > neg_prob):
        correct_pred_count += 1
                      
for review in cleaned_test_neg:
    pos_prob = 1
    neg_prob = 1
    for word in review:
        pos_likelihood = (1 / (pos_word_count + vocab_size))
        if word in pos_word_freq:
            pos_likelihood = ((pos_word_freq[word] + 1) / (pos_word_count + vocab_size))
        pos_prob *= pos_likelihood
        neg_likelihood = (1 / (neg_word_count + vocab_size))
        if word in neg_word_freq:
            neg_likelihood = ((neg_word_freq[word] + 1) / (neg_word_count + vocab_size))
        neg_prob *= neg_likelihood
        
    neg_prob *= prior_prob_neg
    pos_prob *= prior_prob_pos

    if (neg_prob > pos_prob):
        correct_pred_count += 1


accuracy = correct_pred_count/500
print(accuracy)





