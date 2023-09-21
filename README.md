# Troll_Detection_AI
Automatic troll detection with sentiment analysis and logistic regression.
3. Implementation overview The code consists of 3 phases. 1st phase is the sentiment classification logistic regression model creation. 
2nd phase is the labeling of troll dataset with the sentiment model. 
3rd phase is the troll classification logistic regression model creation.
1.Sentiment model
1. Firstly, we imported all the relevant libraries and downloaded vocabularies from nltk, such as stopwords and wordnet.
2.  Then we created a method that is responsible for preprocessing the text. This includes removal of URLs, removal of Twitter mentions.
   For tokenization, we used TweetTokenizer which was specifically created for tweet message Tokenization.
    Next, we used lemmatization to reduce the words to their basic form and then removed all the remaining stop words in the message.
    Then all the token were joined back together.
3. For the next steps, we read the csv file containing the sentiment dataset. Replaced the labels, that were representing positive labels as 4, to 1. We applied the preprocessing method on the dataset, and then removed any rows that happened to have nan values.
4. After that we continued by splitting the data to testing and training and used the TfidfVectorizer with n_gram=1,3 to fit and transform the train_data and transform the test data. We also printed out the vocabulary size.
5. The fifth steps, include defining the logistic regression model, starting a timer to measure the learning time, and then using the fit method to train the data.
6. When it is done learning, we use the score function to get hold of the accuracy. We also save the model for later use.
 Then we use the test data to test the model on a dataset It haven’t seen before, and write out the evaluation metrics through Classification report, Confusion Matrix and ROC Curve.


2. Dataset labeling
1.For the first step, we take read merged_dataset.csv file and use the preprocessing method on the content column.
After preprocessing we remove any rows that have nan values.
2. Next, we use the already trained TfidfVectorizer on the dataset to transform it into vector forms.
Then we use the sentiment model to predict the value of each message.
4. A new column is created called sentiment, which will contain the binary sentiment value of the message, then we rename this modified dataset to troll_sentiments_LR_TFIDF.csv
5. Troll classification model
1.Firstly we define a new method called merge, which is responsible for adding a new word (Positive,Negative) to the end of the tweet text, indicating the sentiment of the message.
2. For the second step we do the same things we did on the sentiment dataset, preprocessing, nan removal and an additional empty string removal, then the merge method was used.
3. As the third step, we also removed the sentiment column, since it was not needed anymore, then we fitted and transformed the content data on the TfidfVectorizer again and changed the datatype of the content columns for both test and train to Unicode.
4. The lasts steps include training and an evaluation metric the same was as it was for the
sentiment model.
