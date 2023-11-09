CS 579: Online Social Network Analysis 
------------------------------------------------------------------------------------------------------------

Project II - Fake News Classification

------------------------------------------------------------------------------------------------------------

Group 45 

------------------------------------------------------------------------------------------------------------

Vivekanand Reddy Malipatel (A20524871) 
Mohammed Shoaib (A20512491)

------------------------------------------------------------------------------------------------------------

Steps to Run the program :

1. Run readdata.py to fetch the data from test_original, train_original csv files, preprocess it and Store it to test.csv and train.csv files.
2. Next, Run Each of the Machine Learning model stored in different python files to generate the classification metrics with validation data and Predict the labels on the test data. The predicted test data will best stored in seperate csv files for each model.
3. Next, Run the submissiong.py file to generate the submission.csv file with the model output that generates the best classification.

------------------------------------------------------------------------------------------------------------
Libraries Required :

Pandas, nltk, sklearn, numpy


# Fake News Classification Task

## Overview
In the era of digital news consumption, the proliferation of fake news on social media has become a major issue. The goal of this project is to classify pairs of news articles into categories based on their relationship to each other, specifically whether they agree, disagree, or are unrelated to each other. This is crucial for maintaining the authenticity balance of the news ecosystem and for preventing the spread of misinformation.

## Task Definition
Given the title of a fake news article (A) and the title of a coming news article (B), classify B into one of three categories:

- `agreed`: B discusses the same fake news as A.
- `disagreed`: B refutes the fake news in A.
- `unrelated`: B is unrelated to A.

## File Descriptions
Within the provided dataset, there are three CSV files:

- `train.csv`: Contains training data with labels.
- `test.csv`: Contains test data without labels.
- `sample_submission.csv`: Demonstrates the expected submission format.

## Data Columns
Both training and testing data will include the following columns:

- `id`: The ID of each news pair.
- `tid1`: The ID of fake news title 1.
- `tid2`: The ID of news title 2.
- `title1_en`: The English title of fake news 1.
- `title2_en`: The English title of news 2.
- `label`: (Training data only) Indicates the relationship between the news pair (agreed/disagreed/unrelated).

## Instructions

1. **Data Preparation**: Split the `train.csv` file to create a validation set for model evaluation.
2. **Model Training**: Use the training data to train your classifier.
3. **Evaluation**: Assess the performance of your model using the validation set.
4. **Prediction**: Apply your trained model to the `test.csv` file to predict the labels.

