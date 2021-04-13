from surprise import Dataset
from surprise import KNNBasic
from surprise.model_selection import cross_validate
import pandas as pd
from surprise import Reader
from collections import defaultdict

import json


# Get TOP 5 recommendations for each user
def getTop5ForEachUser(predictions):
    # top k are 5
    k = 5
    top5 = defaultdict(list)
    for userID, movieID, r, predicatedValue, _ in predictions:
        top5[userID].append((movieID, predicatedValue))

    # Get top 5 ratings in desc order of predicated ratings.
    for userID, ratings in top5.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[userID] = ratings[:k]

    return top5


data = pd.read_csv("ratings.csv")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
options = {
    'name': 'cosine',
    'user_based': True
}

cosineSimilarityAlgo = KNNBasic(sim_options=options)

# Get training dataset
trainingDataSet = data.build_full_trainset()

# Get test data which does not include training data
testDataSet = trainingDataSet.build_anti_testset()

# TRAIN THE MODEL
cosineSimilarityAlgo.fit(trainingDataSet)

# Try to get predications for test data
predications = cosineSimilarityAlgo.test(testDataSet)

# Perform validation of the predicated results against the given values. The test dataset is divided into two fold
# to get RMSE more accurate of two fold test data
results = cross_validate(cosineSimilarityAlgo, data, measures=['RMSE'], cv=2, verbose=True)

print("Calc top k")
top_n = getTop5ForEachUser(predications)

# Output top5 for each user to JSON
jsonMatrix = {}
for userID, ratings in top_n.items():
    jsonMatrix[str(userID)] = [iid for (iid, _) in ratings]

# Write JSON Obj to the json file
with open('cosineSimilarity.json', 'w') as outfile:
    json.dump(jsonMatrix, outfile)
