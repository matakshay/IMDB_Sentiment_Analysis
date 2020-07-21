<h1>IMDB Sentiment Analysis Model</h1>

<p align="center">
<img src="https://img.shields.io/badge/made%20by%20-matakshay-blue">
<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
<img src="https://img.shields.io/badge/contributions-welcome-brightgreen">
<img src="https://img.shields.io/badge/python-v3.7%2B-orange" />
<img src="https://img.shields.io/badge/tensorflow-2.1.0-yellow">
</p>

<p align="justify">
This is a Sentiment Analysis Model built using Machine Learning and Deep Learning to classify movie reviews from the IMDB dataset into "positive" and "negative" classes.
</p>

<h3> TABLE OF CONTENTS </h3>
<ol type="I">
    <li><a href="#intro"> Introduction </a></li>
    <li><a href="#dataset"> Dataset </a></li>
    <li><a href="#models"> Models </a></li>
        <ol type="i">
            <li><a href="#lstm"> Long Short-Term Memory Network </a></li>
            <li><a href="#convnet"> Convolutional Network </a></li>
        </ol>
    </li>
    <li><a href="#frameworks"> Frameworks, Libraries & Languages </a></li>
    <li><a href="#usage"> Usage </a></li>
    <li><a href="#acknowledgement"> Acknowledgement </a></li>
</ol>

<h2 id="intro">Introduction</h2>

<p align="justify">
Sentiment Analysis has been a classic field of research in Natural Language Processing, Text Analysis and Linguistics. It essentially attempts to identify, categorize and possibly quantify, the opinions expressed in a piece of text and determine the author's attitude toward a topic, product or situation. This has widespread application in Recommender systems for predicting the preferences of users and in e-commerce websites to analyse customer feedback & reviews. Based on the sentiments extracted from the data, companies can better understand their customers and align their businesses accordingly. <br>
Before the advent of the Deep Learning era, Statistical methods and Machine Learning techniques found ample usage for Sentiment Analysis tasks. With the increase in the size of datasets and text corpora available on the internet, coupled with advancements in GPUs and computational power available for these tasks, Neural Networks have ushered in and vastly improved the state-of-the-art performance in various NLP tasks, and Sentiment Analysis remains no exception to this. Recurrent Neural Networks (RNN), Gated RNNs, Long-Short Term Memory networks (LSTM) and 1D ConvNets are some classic examples of neural architectures which have been successful in NLP tasks.  
</p>

<h2 id="dataset"> Dataset </h2>
<p align="justify">
This project uses the <a href="https://ai.stanford.edu/~amaas/data/sentiment/">Large Movie Review Dataset</a> which has been in-built with Keras. This dataset contains 25000 highly polar movie reviews for training, and another 25000 reviews for testing. It does not contain more than 30 reviews for any single movie, and also ensures there are equal number of positive and negative reviews in the both the training and test sets. Additionally, neutral reviews (those with rating 5/10 or 6/10) have been excluded.  
This dataset has been a benchmark for many Sentiment Analysis tasks, since it was first released in 2011.
</p>

<h2 id="models">Models</h2>
<p align="justify">
    I built and experimented with different models to compare their performance on the dataset -
</p>
<ul>
    <li>
        <h4 id="lstm"><b> Long Short-Term Memory Network: </b></h4>
        <p align="justify">
            Recurrent Neural Networks are especially suited for sequential data (sequence of words in this case). Unlike the more common feed-forward neural networks, an RNN does not input an entire example in one go. Instead, it processes a sequence element-by-element, at each step incorporating new data with the information processed so far. This is quite similar to the way humans too process sentences - we read a sentence word-by-word in order, at each step processing a new word and incorporating it with the meaning of the words read so far.
        </p>
        <div align="center">
            <figure>
                <img src = "RNN.png"
                     alt = "RNN Diagram"
                     width = 500>
                <figcaption>
                    A diagram of a recurrent neural network
                </figcaption>
            </figure>
        </div>
        <div align="center">
            <figure>
                <img src = "LSTM.png"
                     alt = "RNN Diagram"
                     width = 500>
                <figcaption>
                    A diagram of an LSTM network.
                </figcaption>
            </figure>
        </div>
        <p align="justify">
            LSTMs further improve upon these vanilla RNNs. Although theoretically RNNs are able to retain information over many time-steps ago, practically it becomes extremely difficult for simple RNNs to learn long-term dependencies, especially in extremely long sentences and paragraphs. LSTMs have been designed to have special mechanisms to allow past information to be reutilised at a later time. As a result, in practise, LSTMs are almost always preferable over vanilla RNNs. <br >
            Here, I built an LSTM model using Keras Sequential API. A summary of the model and its layers is given below. The model was trained with a batch size of 64, using the Adam Optimizer.
        </p>
        <div align="center">
            <figure>
                <img src = "LSTM_model_visual.png"
                     alt = "LSTM Model Visual"
                     width = 400>
                <figcaption> A plot of the model and its layers </figcaption>
            </figure>
        </div>
        <p align="justify">
            While tuning the hyper-parameters, a Dropout layer was introduced as measure of regularization to minimize the overfitting of the model on the training dataset. A separate validation set (taken from the training data) was used to check the performance of the model during this phase. <br>
            This model managed to achieve an accuracy of <b>85.91%</b> when evaluated on the hidden test dataset (and <b>99.96%</b> on the training dataset).
        </p>
    </li>
    <li>
        <b><h4 id="convnet">Convolutional Network:</h4></b>
        <p align="justify">
            The idea of Convolutional Networks has been quite common in Computer Vision. The use of convolutional filters to extract features and information from pixels of an image allows the model to identify edges, colour gradients, and even specific features of the image like positions of eyes & nose (for face images). Apart from this, 1D Convolutional Neural Networks have also proven quite competitive with RNNs for NLP tasks. Given a sequential input, 1D CNNs are well able to recognize and extract local patterns in this sequence. Since the same input transformation is performed at every patch, a pattern learned at a certain position in the sequence can very easily later be recognized at a differnt position.
            Further, in comparison to RNNs, ConvNets in general are extremely cheap to train computationally - In the current project (built using Google Colaboratory with a GPU kernel), the LSTM model took more than 30 minutes to complete an epoch (during training) while the CNN model took hardly 9 seconds on average!
            <br>   
            I built the model using Keras Sequential API. A summary of the model and its layers is below.
        </p>
        <div align="center">
            <figure>
                <img src = "CNN_model_visual.png"
                     alt = "CNN Model Visual"
                     width = 500>
                <figcaption> A plot of the model and its layers </figcaption>
            </figure>
        </div>
        <p align="justify">
            This model was trained with a batch size of 64 using Adam Optimizer. The best model (weights and the architecture) was saved during this phase.
            This model achieved an accuracy of <b>89.7 %</b> on the test dataset, a good increase over the LSTM model.
        </p>
    </li>
</ul>

<h2 id="frameworks">Frameworks, Libraries & Languages</h2>
<ul>
    <li> Keras </li>
    <li> Tensorflow </li>
    <li> Python3 </li>
    <li> Matplotlib </li>
</ul>

<h2 id="usage">Usage</h2>
<p align="justify"> On the terminal run the following commands- </p>
    <ol>
        <li>
            Install all dependencies
            <br>
            <code> pip install python3 </code>
            <br>
            <code> pip install matplotlib </code>
            <br>
            <code> pip install tensorflow </code>
            <br>
            <code> pip install keras </code>
        </li>
        <li>
            Clone this repository on your system and head over to it
            <br>
              <code> git clone https://github.com/matakshay/IMDB_Sentiment_Analysis </code>
            <br>
            <code size> cd IMDB_Sentiment_Analysis </code>
        </li>
        <li>
            Either of the CNN or LSTM model can be used to predict for a custom movie review. <br>
            To run the LSTM model -
            <br>
                <code> python3 LSTM_predict.py </code>
            <br>
            This loads the LSTM model with its weights and prompts for an input.       
            <br><br>
            To run the CNN model -
            <br>
                <code> python3 CNN_predict.py </code>
            <br>
            This loads the CNN model with its weights and prompts for an input.
        </li>
        <li>
            Type a movie review (in English) in the terminal and get its sentiment class predicted by the model
        </li>
    </ol>
    
<h2 id="acknowledgement">Acknowledgement</h2>
<p align="justify">
    I studied and referred many articles, books and research papers while working on this project. I am especially grateful to the authors of the following for their work -
</p>
    <ul>
        <li> https://colah.github.io/posts/2015-08-Understanding-LSTMs/ </li>
        <li> https://medium.com/@romannempyre/sentiment-analysis-using-1d-convolutional-neural-networks-part-1-f8b6316489a2 </li>
        <li> Deep Learning with Python by François Chollet </li>
    </ul>
<p align="justify">
    Some other websites I referred -
</p>
<ul>
    <li> https://keras.io/ </li>
    <li> https://en.wikipedia.org/wiki/Naive_Bayes_classifier </li>
    <li> https://en.wikipedia.org/wiki/Sentiment_analysis </li>
</ul>
