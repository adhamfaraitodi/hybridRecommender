# -*- coding: utf-8 -*-
Dataset rekomendasi buku resep makanan

import dataset dan library
"""

import pandas as pd
import numpy as np

df = pd.read_excel('.xlsx')
print(df.head())

!pip install scikit-surprise
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# for cosine similarity using linear_kernel
# content_df = df[['recipe_id', 'recipe_name']]
# content_df['Content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

# # Use TF-IDF vectorizer to convert content into a matrix of TF-IDF features
# tfidf_vectorizer = TfidfVectorizer()
# content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])

# content_similarity = linear_kernel(content_matrix, content_matrix)

# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(df[['user_id',
#                                   'recipe_id',
#                                   'rating']], reader)

# def get_content_based_recommendations(product_id, top_n):
#     index = content_df[content_df['recipe_id'] == product_id].index[0]
#     similarity_scores = content_similarity[index]
#     similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
#     recommendations = content_df.loc[similar_indices, 'recipe_id'].values
#     return recommendations

# for Pearson correlation coefficient from library sklearn
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

content_similarity = 1 - linear_kernel(content_matrix, content_matrix)

def get_content_based_recommendations(product_id, top_n):
    index = content_df[content_df['recipe_id'] == product_id].index[0]
    similarity_scores = content_similarity[index]
    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
    recommendations = content_df.loc[similar_indices, 'recipe_id'].values
    return recommendations

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

def get_hybrid_recommendations(user_id, product_id, top_n):
    content_based_recommendations = get_content_based_recommendations(product_id, top_n)
    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id, top_n)
    # Combine the recommendations from both approaches
    all_recommendations = np.concatenate([content_based_recommendations, collaborative_filtering_recommendations])
    # Get unique product IDs using numpy.unique
    hybrid_recommendations_ids = np.unique(all_recommendations)
    # Convert unique product IDs to product names
    hybrid_recommendations = content_df[content_df['recipe_id'].isin(hybrid_recommendations_ids)].drop_duplicates(subset=['recipe_id'])
    # Extract the desired columns
    hybrid_recommendations = hybrid_recommendations[['recipe_id', 'recipe_name']]

    return hybrid_recommendations[:top_n]

user_id = 1
product_id = 229875
top_n = 10
recommendations = get_hybrid_recommendations(user_id, product_id, top_n)

print(f"Hybrid Recommendations for User {user_id} based on Product {product_id}:")
# print(recommendations)
for i, row in recommendations.iterrows():
    print(f"recipe_id: {row['recipe_id']}, Product Name: {row['recipe_name']}")