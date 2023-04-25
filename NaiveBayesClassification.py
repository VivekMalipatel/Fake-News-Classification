from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train, test = train_test_split(train_df, test_size=0.3,random_state= 62)

# Define TF-IDF vectorizer with custom parameters
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')

# Fit vectorizer on preprocessed training data
X_train = vectorizer.fit_transform(train['title1_en_clean'] + ' ' + train['title2_en_clean'])

# Transform preprocessed test data using the fitted vectorizer
X_train_test = vectorizer.transform(test['title1_en_clean'] + ' ' + test['title2_en_clean'])
X_test = vectorizer.transform(test_df['title1_en_clean'] + ' ' + test_df['title2_en_clean'])

Y_train = train['label']
Y_train_test = test['label']


# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, Y_train)

y_pred_train = nb_classifier.predict(X_train_test)
print(classification_report(Y_train_test, y_pred_train))

accuracy = accuracy_score(Y_train_test, y_pred_train)
print("Model Accuracy: "+ str(round(accuracy*100,2))+ "%")

# Evaluate classifier on validation set
y_pred = nb_classifier.predict(X_test)

NBpredictedtest = pd.concat([test_df[['id','tid1','tid2','title1_en','title2_en']], pd.DataFrame({'predictred_label': y_pred})], axis=1)
NBpredictedtest.to_csv('NBpredictedtest.csv')

print("71.68")