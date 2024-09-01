import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
from flask import Flask, request, jsonify, render_template

# Load the dataset
file_path = r'C:\Users\Me\Downloads\flipkart_reviews_dataset.csv' 
data = pd.read_csv(file_path)

# Data Cleaning Function
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Apply text cleaning
data['cleaned_reviews'] = data['summary'].apply(clean_text)  # Assuming 'summary' is the review text

# Assign sentiments based on ratings
data['sentiment'] = np.where(data['rating'] > 3, 'positive', 
                             np.where(data['rating'] == 3, 'neutral', 'negative'))

# Balance the dataset
positive_reviews = data[data['sentiment'] == 'positive']
neutral_reviews = data[data['sentiment'] == 'neutral']
negative_reviews = data[data['sentiment'] == 'negative']

# Find the minimum number of samples in any class
min_count = min(len(positive_reviews), len(neutral_reviews), len(negative_reviews))

# Downsample each class to the minimum count
balanced_data = pd.concat([
    positive_reviews.sample(min_count, random_state=42),
    neutral_reviews.sample(min_count, random_state=42),
    negative_reviews.sample(min_count, random_state=42)
])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into features and labels
X = balanced_data['cleaned_reviews']
y = balanced_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Machine Learning Model
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
model.fit(X_train, y_train)

# Save the model
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']  # Get the review from the form
    cleaned_review = clean_text(review)
    prediction = model.predict([cleaned_review])
    return render_template('index.html', review=review, sentiment=prediction[0])  # Return the result to the HTML template

if __name__ == '__main__':
    app.run(debug=True)

