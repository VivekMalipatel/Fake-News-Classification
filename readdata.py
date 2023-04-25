import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Load the train and test CSV files
train_df = pd.read_csv('train_original.csv')
test_df = pd.read_csv('test_original.csv')

# Check for missing values, duplicates, and any other data quality issues
print(train_df.isnull().sum()) # check for missing values in train data
print(test_df.duplicated().sum()) # check for duplicates in test data

# Explore the data and gain a better understanding of the features
print(train_df.describe()) # summary statistics for numeric features
print(train_df['label'].value_counts()) # count of labels in train data

# Preprocess the text data
stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english')

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = text.lower() # convert to lowercase
    words = text.split()
    words = [w for w in words if not w in stop_words] # remove stopwords
    words = [stemmer.stem(w) for w in words] # stem the words
    text = ' '.join(words)
    return text

train_df['title1_en_clean'] = train_df['title1_en'].apply(clean_text)
train_df['title2_en_clean'] = train_df['title2_en'].apply(clean_text)
train_df = train_df[(train_df['title1_en_clean'].str.strip() != '') & (train_df['title2_en_clean'].str.strip() != '')]
train_df.to_csv('train.csv')
test_df['title1_en_clean'] = test_df['title1_en'].apply(clean_text)
test_df['title2_en_clean'] = test_df['title2_en'].apply(clean_text)
test_df = test_df[(test_df['title1_en_clean'].str.strip() != '') & (test_df['title2_en_clean'].str.strip() != '')]
test_df.to_csv('test.csv')