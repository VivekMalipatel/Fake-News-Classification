import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load the train and test CSV files with preprocessed text data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Combine the preprocessed text features
train_df['combined'] = train_df['title1_en_clean'] + ' ' + train_df['title2_en_clean']
test_df['combined'] = test_df['title1_en_clean'] + ' ' + test_df['title2_en_clean']
train_df.dropna(subset=['combined'], inplace=True)
test_df.dropna(subset=['combined'], inplace=True)

# Create the count vectorizer and fit on the combined text features
count_vect = CountVectorizer()
count_vect.fit(train_df['combined'])

# Create the TF-IDF vectorizer and fit on the combined text features
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(train_df['combined'])

# Create the bag-of-words features using the count vectorizer
train_count_features = count_vect.transform(train_df['combined'])
test_count_features = count_vect.transform(test_df['combined'])

# Create the TF-IDF features using the TF-IDF vectorizer
train_tfidf_features = tfidf_vect.transform(train_df['combined'])
test_tfidf_features = tfidf_vect.transform(test_df['combined'])

# Scale the numeric features
scaler = StandardScaler()
scaler.fit(train_df[['tid1', 'tid2']])
train_numeric_features = scaler.transform(train_df[['tid1', 'tid2']])
test_numeric_features = scaler.transform(test_df[['tid1', 'tid2']])

# Combine the features
train_features = np.hstack((train_count_features.toarray(), train_tfidf_features.toarray(), train_numeric_features))
test_features = np.hstack((test_count_features.toarray(), test_tfidf_features.toarray(), test_numeric_features))
