import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load the train and test CSV files with preprocessed text data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

def create_features(df):

    df.dropna(inplace=True)
    
    # Define the vectorizers
    count_vec = CountVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000)
    tfidf_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000)
    
    # Create bag-of-words features
    title1_count = count_vec.fit_transform(df['title1_en_clean'])
    title2_count = count_vec.transform(df['title2_en_clean'])
    title1_tfidf = tfidf_vec.fit_transform(df['title1_en_clean'])
    title2_tfidf = tfidf_vec.transform(df['title2_en_clean'])
    
    # Compute additional features
    title1_len = df['title1_en_clean'].apply(len)
    title2_len = df['title2_en_clean'].apply(len)
    len_diff = (title1_len - title2_len).apply(abs)
    len_ratio = (np.maximum(title1_len, title2_len) / np.minimum(title1_len, title2_len))
    
    # Combine the features into a single array
    features = np.hstack((title1_count.toarray(), title2_count.toarray(),
                          title1_tfidf.toarray(), title2_tfidf.toarray(),
                          title1_len.values.reshape(-1,1), title2_len.values.reshape(-1,1),
                          len_diff.values.reshape(-1,1), len_ratio.values.reshape(-1,1)))
    
    return features

train_features = create_features(train_df)
test_features = create_features(test_df)
