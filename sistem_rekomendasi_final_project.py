# -*- coding: utf-8 -*-
"""
import dataset dan library
"""

!pip install scikit-surprise

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from surprise import BaselineOnly
from surprise.model_selection import cross_validate, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_excel('.xlsx')
print(df.head())

content_df = df[['recipe_id', 'recipe_name']]
content_df['Content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)


tfidf_vectorizer = TfidfVectorizer()
content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])
from sklearn.metrics.pairwise import euclidean_distances
content_distance = euclidean_distances(content_matrix)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)

def get_content_based_recommendations(product_id, top_n):
    index = content_df[content_df['recipe_id'] == product_id].index[0]
    distance_scores = content_distance[index]
    similar_indices = distance_scores.argsort()[:top_n + 1]
    recommendations = content_df.loc[similar_indices, 'recipe_id'].values
    return recommendations

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=.25)
all_predictions = []


for user_id, product_id, true_rating in testset:
    content_based_recommendations = get_content_based_recommendations(product_id, top_n=10)
    if product_id in content_based_recommendations:
        predicted_rating = 5.0
    else:
        predicted_rating = 1.0
    all_predictions.append((user_id, product_id, true_rating, predicted_rating,None))
accuracy.rmse(all_predictions)
accuracy.mae(all_predictions)

algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

def get_collaborative_filtering_recommendations(user_id, top_n):
    testset = trainset.build_anti_testset()
    testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [prediction.iid for prediction in predictions[:top_n]]
    return recommendations

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)

bsl_options = {'method': 'sgd', 'learning_rate': .00005,}
algo = BaselineOnly(bsl_options=bsl_options)


algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

from collections import Counter

def get_hybrid_recommendations(user_id, product_id, top_n):
    content_based_recommendations = get_content_based_recommendations(product_id, top_n)
    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id, top_n)

    all_recommendations = list(content_based_recommendations) + list(collaborative_filtering_recommendations)

    recommendation_counts = Counter(all_recommendations)
    hybrid_recommendations = sorted(recommendation_counts, key=lambda x: (-recommendation_counts[x], all_recommendations.index(x)))
    hybrid_recommendations = hybrid_recommendations[:top_n]

    hybrid_recommendations = content_df[content_df['recipe_id'].isin(hybrid_recommendations)].drop_duplicates(subset=['recipe_id'])
    hybrid_recommendations = hybrid_recommendations[['recipe_id', 'recipe_name']]

    return hybrid_recommendations

user_id = 1
product_id = 229875
top_n = 10
recommendations = get_hybrid_recommendations(user_id, product_id, top_n)

print(f"Hybrid Recommendations for User {user_id} based on Product {product_id}:")
for i, row in recommendations.iterrows():
    print(f"recipe_id: {row['recipe_id']}, Product Name: {row['recipe_name']}")