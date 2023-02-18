# Python code for sentiment analysis and topic modeling using machine learning and NLP

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('customer_feedback.csv')

# Preprocess the data using NLP techniques
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(df['feedback'].values)

# Train a Latent Dirichlet Allocation model on the data to identify common topics
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)

# Identify the most important words for each topic
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx+1))
    print(" ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-10 - 1:-1]]))

# Train a machine learning model on the data to classify feedback as positive, negative, or neutral
y = df['sentiment'].values
clf = LogisticRegression()
clf.fit(X, y)

# Evaluate the accuracy of the model
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print('Accuracy:', accuracy)
