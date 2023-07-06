# SMS-Spam-Classification-Using-Machine-Learning

## Aim:
The aim of this project is to build a machine learning model that can accurately classify SMS messages as either spam or ham (non-spam). The project involves data preprocessing, exploratory data analysis (EDA), and model building using various classifiers. The goal is to identify and evaluate the best performing classifier for this task.

## Description:
In this project, we start by loading the SMS spam collection dataset and performing necessary data preprocessing steps. This includes splitting the dataset into the category (spam or ham) and the corresponding text messages. We then perform exploratory data analysis to gain insights into the dataset, such as visualizing the class distribution and analyzing the length of messages, number of words, and number of sentences in each SMS.

Next, we preprocess the text data by converting it to lowercase, tokenizing it into words, removing special characters, stopwords, and punctuation, and performing stemming. We transform the text data using these preprocessing steps to create a feature vector.

We further analyze the preprocessed data by visualizing word clouds for both spam and ham messages to identify frequently occurring words. This helps us understand the key features that differentiate spam and ham messages.

To build the classification models, we utilize two approaches: CountVectorizer and TF-IDF vectorizer. We split the preprocessed data into training and testing sets and train several classifiers using both vectorization techniques. The classifiers include Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Support Vector Machines, K-Nearest Neighbors, Decision Trees, Logistic Regression, Random Forests, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost.

We evaluate the classifiers based on their accuracy and precision scores using both vectorization techniques. The models are compared to identify the best performing classifier for the SMS spam classification task.

## Conclusion:
In this project, we successfully built and evaluated multiple machine learning models for SMS spam classification. After preprocessing the data and transforming it using either CountVectorizer or TF-IDF vectorizer, we trained and tested various classifiers. Based on the evaluation results, the Multinomial Naive Bayes classifier with TF-IDF vectorization demonstrated the best performance, achieving high precision in identifying spam messages.

The project highlights the importance of preprocessing text data and the effectiveness of machine learning models in classifying SMS messages. By accurately identifying spam messages, this classification system can help users filter unwanted messages, reduce potential risks, and enhance overall messaging experience.
