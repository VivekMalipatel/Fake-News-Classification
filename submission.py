import pandas as pd

test_df = pd.read_csv('test_original.csv')
modelprediction = pd.read_csv('SVMpredictedtest.csv')
submission = modelprediction[['id','predictred_label']].rename(columns={'predictred_label':'label'})
not_in_df2 = test_df[~test_df['id'].isin(submission['id'])]
print("Number of rows removed from Test Data, removed in filtering due to improper data : ",not_in_df2.shape[0])
print("Test Data Rows removed in filtering due to improper data : ")
print(not_in_df2)
submission.set_index('id',inplace=True)
submission.to_csv('submission.csv')

