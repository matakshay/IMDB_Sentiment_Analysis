# Loads the CNN model and uses it to predict the Sentiment class of input moview review
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

model = load_model ('CNN_model/CNN_model.hdf5')

print("Type a movie review :")
input_review = str(input())

word_to_index = imdb.get_word_index()
print("Processing input...")
tokenized_review = [word.lower() for word in input_review.split()]
text_indices = [word_to_index[word]+3 for word in tokenized_review if word in word_to_index.keys()]
model_input = [text_indices]
model_input = pad_sequences (model_input, maxlen = 2494)

print("Running the model with input...")
prediction = model.predict(model_input, verbose = 0)

if (prediction>0.5):
    print("Prediction: Positive")
else:
    print("Prediction: Negative")
