# Import Libraries
from newspaper import Article
import nltk
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
import ssl
import nltk

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt_tab')
nltk.download('punkt')
warnings.filterwarnings('ignore')

# Download 'punkt' package
nltk.download('punkt', quiet=True)

# Input Article
article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()

corpus = article.text

# Tokenization
text = corpus
sentence_list = nltk.sent_tokenize(text)  # A list of sentences
print(sentence_list)
# Create bot response
def bot_response(user_input):
    user_input = user_input.lower()
    sentence_list.append(user_input)

    count_vect = CountVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(count_vect[-1], count_vect)
    similarity_scores_list = similarity_scores.flatten()
    index = similarity_scores_list.argsort()[::-1]

    response_flag = 0
    bot_response = ''

    for i in range(1, len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            bot_response = bot_response + ' ' + sentence_list[index[i]]
            response_flag = 1
            if i > 2:
                break

    if response_flag == 0:
        bot_response = bot_response + " I apologize, I don't understand."

    sentence_list.remove(user_input)

    return bot_response


# Function to return random greeting response
def greeting_response(text):
    text = text.lower()

    # Bot's greeting responses
    bot_greetings = ['hi', 'hey', 'hello', 'hola']

    # User's greeting inputs
    user_greetings = ['hi', 'hey', 'hello', 'hola', 'greetings']

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)


# Start chatbot
print("Doc Bot: I am Doc Bot. I'm here to answer your questions about chronic kidney disease. Type 'exit' to leave.")

exit_list = ['exit', 'see you later', 'bye', 'quit', 'break']

while True:
    user_input = input()
    if user_input.lower() in exit_list:
        print("Doc Bot: Chat with you later!")
        break
    else:
        if greeting_response(user_input) is not None:
            print("Doc Bot: " + greeting_response(user_input))
        else:
            print("Doc Bot:" + bot_response(user_input))