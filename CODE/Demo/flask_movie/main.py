import json

from flask import Flask, render_template, request
from flask_fontawesome import FontAwesome
import pandas as pd
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
import warnings
from csv import reader

warnings.filterwarnings("ignore")
app = Flask(__name__)
fa = FontAwesome(app)


def predictMovies(userid):
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    data = pd.merge(ratings, movies, how="inner")
    data['movieId'] = data['movieId'].astype(str)

    Word2VecModel = Word2Vec.load("Word2VecModel.model")
    model = load_model('finalLSTMModel.h5')

    userData = data.loc[data['userId'] == userid]
    userData = userData.sort_values('timestamp').reset_index()
    vocab = Word2VecModel.wv.vocab
    xuser = []
    finxuser = []
    for i in range(len(userData)):
        if (userData.iloc[i].movieId in vocab):
            #             print(userData.iloc[i].movieId)
            xuser.append(Word2VecModel[userData.iloc[i].movieId])
    finxuser = xuser[-5:]
    topred = []
    topred.append(finxuser)
    topred = np.array(topred, dtype=np.float32)
    ans = np.argmax(model.predict(topred)).astype(str)
    return ans, Word2VecModel


def similarMovies(movieId, n, Word2VecModel):
    # most similar movies
    similarMovies = Word2VecModel.similar_by_vector(movieId, topn=n + 1)[1:]

    #  list Of Similar Movies 
    listOfSimilarMovies = []
    listOfSimilarMovies.append(movieId)
    for i in similarMovies:
        listOfSimilarMovies.append(i[0])

    listOfSimilarMovies = [int(i) for i in listOfSimilarMovies]

    return listOfSimilarMovies


def getRecommendations(userId):
    movie, Word2VecModel = predictMovies(userId)
    listOfSimilarMovies = similarMovies(movie, 4, Word2VecModel)
    movies = pd.read_csv('movies.csv')
    jsonResponse = {
        listOfSimilarMovies[0]: movies.loc[movies['movieId'] == listOfSimilarMovies[0], 'title'].iloc[0],
        listOfSimilarMovies[1]: movies.loc[movies['movieId'] == listOfSimilarMovies[1], 'title'].iloc[0],
        listOfSimilarMovies[2]: movies.loc[movies['movieId'] == listOfSimilarMovies[2], 'title'].iloc[0],
        listOfSimilarMovies[3]: movies.loc[movies['movieId'] == listOfSimilarMovies[3], 'title'].iloc[0],
        listOfSimilarMovies[4]: movies.loc[movies['movieId'] == listOfSimilarMovies[4], 'title'].iloc[0]
    }

    print(jsonResponse)
    return jsonResponse


def getMFRecommendations(userID):
    predicted_ratings = np.genfromtxt('final_ratings.csv', delimiter=',')
    N = 5
    res = sorted(range(len(predicted_ratings[userID])), key=lambda sub: (predicted_ratings[userID])[sub])[-N:]
    result = {}

    for i in range((len(res) - 1), (len(res) - 6), -1):
        with open('movies.csv', 'r', encoding='utf-8') as movies:
            r = reader(movies)
            print(res[i])
            for row in r:
                if row[0] == str(res[i]):
                    print(row[1]);
                    result[res[i]] = row[1]
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommendationsLSTM")
def recommendationsLSTM():
    userID = request.args.get('userID')
    userID = int(userID)
    print(type(userID))
    ans = getRecommendations(userID)
    return ans;


@app.route("/recommendationsMF")
def recommendationsMF():
    userID = request.args.get('userID')
    userID = int(userID)
    print(type(userID))
    ans = getMFRecommendations(userID)
    jsonResponse = {6: "Heat (1995)", 555: "True Romance (1993)", 2353: "Enemy of the State (1998)",
                    3218: "Poison (1991)", 3596: "Screwed (2000)"}
    return ans


@app.route("/recommendationsCosine")
def recommendationsCosine():
    userID = request.args.get('userID')
    userID = str(userID)
    predictions = json.load(open("cosineSimilarity.json"))

    data = pd.read_csv("movies.csv")

    jsonResponse = {}
    i = 0
    for itr in predictions[userID]:
        for index, row in data.iterrows():
            if row['movieId'] == itr:
                jsonResponse[i] = row['title']
                break
        i += 1
    return jsonResponse


if __name__ == "__main__":
    app.run(debug=True)
