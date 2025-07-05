---
layout: post
title: Sentiment-Enhanced Hybrid Recommendation System
image: "/posts/sentiments-image.png"
tags: [Recommendation System, TensorFlow]
---

I analyzed customer sentiments and built a personalized recommendation system that integrates sentiment analysis of customer reviews into the recommendation logic to enhance personalization and recommendation quality, leading to a better user experience. This system combines:by combining 
- Collaborative filtering (via Surprise's SVD) to capture user–item interaction patterns
- Content-Based Filtering using product metadata (e.g., category, brand, etc.); and
- Custom sentiment classification with a deep learning model (LSTM with Bidirectional layers) trained on raw review text.

By integrating sentiment analysis into a hybrid recommender system, I demonstrated how NLP can be leveraged beyond classification to directly enhance recommendation quality and drive smarter, emotionally-informed personalization.

**Business Impact**

By incorporating real-time sentiment insights into the recommendation logic, the system:
- Avoids recommending poorly reviewed products even if they are frequently bought.
- Improves customer trust and engagement by aligning suggestions with emotional satisfaction of past customers, therefore users are more likely to trust the suggestions they receive.
- Encourages high-quality purchases, likely increasing conversion rates and long-term customer loyalty.
- Differentiates user experience, giving the business a competitive edge in product personalization.

---
# **Sentiment Prediction from Customer Reviews**

To understand customer opinion and enhance recommendation relevance, I trained a custom sentiment classification model using customer reviews.

I used TensorFlow's TextVectorization layer for text preprocessing, Embedding layer for word representation, Bidirectional(LSTM) for context learning, and Dense, Dropout, final softmax classification layers.

Output:

- Trained model to classify review sentiment from raw text reviews
- Used model to predict sentiment for all reviews
- Mapped predictions to sentiment probability scores (0–1)
- Aggregated review sentiment per product for use in the recommendation engine

Import the required libraries

```ruby
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras import Sequential, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
```
First I visualized the distribution of sentiments and rating. 

```ruby
sns.countplot(df, x="rating")

percentages = df['sentiment'].value_counts(normalize=True) * 100
percentages = percentages.round(2) 

explode = [0.1 if sentiment == 'negative' else 0 for sentiment in percentages.index]

# Plot the exploded pie chart
plt.figure(figsize=(5, 5))
plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140,
     colors=['#66b447', '#be9c34', '#a34f9e'], explode=explode)
plt.title('Sentiment Distribution with Negative Exploded')
plt.axis('equal')  # Ensures the pie is drawn as a circle.
plt.savefig("sentiment_dist.png")
plt.show()
```
![alt text](/img/sentiment_dist.png "pie chart")

## Preprocessing

I label encoded the target "sentiments", mapped the review text to integer sequences, and converted inputs to numpy.

```ruby
label_encoder = LabelEncoder()
df.loc[:, 'sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['review'].values, df['sentiment_encoded'].values,  
    test_size = 0.2, random_state = 42, stratify = df['sentiment_encoded'])

# Text Vectorization
MAX_VOCAB = 5000
MAX_LEN = 70

vectorizer = TextVectorization(max_tokens=MAX_VOCAB, output_mode='int', output_sequence_length=MAX_LEN)
vectorizer.adapt(X_train)

# Apply the vectorizer to the text data
X_train_vec = vectorizer(X_train)
X_test_vec = vectorizer(X_test)

# convert inputs to numpy
X_train_vec_np = X_train_vec.numpy()
X_test_vec_np = X_test_vec.numpy()
```

I then build the tensorflow sequential model with softmax activation in the last layer to output probabilities.

```ruby
model = Sequential([
    Embedding(input_dim = MAX_VOCAB, output_dim = 256, input_length = MAX_LEN),
    Bidirectional(LSTM(128)),
    Dropout(0.5),
    Dense(256, activation = 'relu'),
    Dropout(0.5),
    Dense(64, activation = 'relu'),
    Dropout(0.3),
    Dense(3, activation = 'softmax')
])
```
Next I compiled the model with adam optimizer and sparse_categorical_crossentropy as the loss function.

```ruby
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```
I then set up callbacks for the training to monitor validation loss and stop after 3 epochs with no improvement in validation loss, and set lower validation loss is better.

```ruby
# Setup Callbacks
callback = EarlyStopping(monitor = 'val_loss', patience = 3,
                         mode = 'min', restore_best_weights = True)

checkpoint = ModelCheckpoint('best_model.keras', monitor ='val_loss', save_best_only = True,
                             mode = 'min', verbose = 1)
```

Lastly I trained and evaluated the model using classification report, confusion matrix and validation accuracy.

```ruby
#fit and train the model

history = model.fit(X_train_vec.numpy(), y_train, validation_data = (X_test_vec_np, y_test), epochs = 15,
    batch_size = 32, validation_batch_size = 32, callbacks = [callback, checkpoint])

# Evaluate the Model
test_loss, test_acc, = model.evaluate(X_test_vec_np, y_test)
print(f"\nTest Accuracy: {test_acc:.2f}")

# inspect metrics

y_pred = model.predict(X_test_vec_np)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

labels = ['Negative', 'Neutral', 'Positive']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```
![alt text](/img/cm_cr_sentiment.png "confusion matrix")

The sentiment classification model achieved:
- Overall Accuracy: 99% on a test set of ~186,000 samples
- F1-Score: 0.99 for Positive and Neutral classes; 0.90 for Negative.
- Confusion Matrix showed excellent class separation, especially for Neutral and Positive classes. 

This high performance enabled the reliable transformation of textual reviews into quantitative sentiment signals for use in recommendations.

# **Sentiment-Enhanced Hybrid Recommender's System**

I integrated the sentiments into a hybrid recommender system, directly enhancing recommendation quality and driving smarter, emotionally-informed personalization.

After training the sentiment classifier, I mapped the predicted sentiments to normalized sentiment scores (0–1) and aggregated them per product. I then used these scores as a third signal in the hybrid recommender alongside collaborative and content scores.

```ruby
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import plotly.express as px
```
### Map predictions back to the original df

```ruby
df_sentiment = df.loc[test_idx].copy()

label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
df_sentiment['predicted_sentiment'] = y_pred_labels
df_sentiment['sentiment_label'] = df_sentiment['predicted_sentiment'].map(label_map)
```

### Convert Predicted Probabilities to Sentiment Scores (0–1 scale)

I used the probability that a review is positive as the sentiment score

```ruby
df_sentiment['sentiment_score'] = y_pred_probs[:, 2]
```
### Slice the data for memory purposes

I sliced the first 10k rows 

```ruby
df_shuffled = df_sentiment.sample(frac=1, random_state=42).reset_index(drop=True)
df = df_shuffled.iloc[:10_000]

# Keep only relevant columns for the recommender
df = df[['Customer_ID', 'product_id', 'rating', 'review', 'Category', 'Type', 'Brand', 
         'Product_name', 'sentiment_label', 'predicted_sentiment', 'sentiment_score']]
```
## Collaborative Filtering with Surprise

I first normalized the predicted sentiments and boosted ratings by sentiment with a weight of 0.3, and then trained a collaborative filtering model to capture user-product interaction patterns. I finally defined a function to predict how a given user is likely to rate a given product. 

```ruby
df['sentiment_norm'] = (df['predicted_sentiment'] - df['predicted_sentiment'].min()) /
                           (df['predicted_sentiment'].max() - df['predicted_sentiment'].min())

df['adjusted_rating'] = df['rating'] * (0.7 + 0.3 * df['sentiment_norm'])
```
```ruby
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['Customer_ID', 'product_id', 'adjusted_rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD model
model_cf = SVD()
model_cf.fit(trainset)
```
```ruby
# Function to predict rating
def predict_cf(user_id, product_id):
    try:
        pred = model_cf.predict(user_id, product_id)
        return pred.est
    except:
        return np.nan
```

## Content-Based Filtering (TF-IDF on Product Info)

After combining product features into a single string, I used scikit-learn's TfidfVectorizer to transform the product information into numerical representation and then computed their cosine similarities. Lastly, I defined function to get top-N similar products. 

```ruby
# Combine product features into a single string
df['product_text'] = df['Category'].fillna('') + ' ' + df['Type'].fillna('') + ' ' + df['Brand'].fillna('') +
                     ' ' + df['Product_name'].fillna('')

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['product_text'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping from product_id to index
product_indices = pd.Series(df.index, index=df['product_id']).drop_duplicates()

# Function to get top-N similar products
def get_content_scores(product_id, top_n=10):
    if product_id not in product_indices:
        return {}
    idx = product_indices.get(product_id)
    # Return empty if not found
    if idx is None or isinstance(idx, pd.Series):
        return {}
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    similar_products = [(df['product_id'].iloc[i], score) for i, score in sim_scores]
    return dict(similar_products)
```
## Hybrid Recommendation Function

I aggregated the sentiment scores (the predicted sentiment probabilities (0 to 1) from classification model above) per product. I then defined a function that pulls the collaborative filtering (CF) score, gets content-based similarity score,  retrieves the average predicted sentiment score for the product, and returns a final weighted hybrid score (combined weights: alpha for CF, 1 - alpha - beta for content, beta for sentimen). 

```ruby
# Average sentiment score per product
product_sentiment = df_sentiment.groupby('product_id')['sentiment_score'].mean().to_dict()

def get_sentiment_score(product_id):
    return product_sentiment.get(product_id, 0.5)

def hybrid_score(user_id, product_id, alpha=0.9, beta=0.005):
    cf_score = predict_cf(user_id, product_id)
    content_scores = get_content_scores(product_id)
    content_score = content_scores.get(product_id, 0)

    # Use precomputed sentiment score
    sentiment_score = get_sentiment_score(product_id)

    # Handle missing values
    if np.isnan(cf_score):
        cf_score = 0

    # Final weighted hybrid score
    return alpha * cf_score + (1 - alpha - beta) * content_score + beta * sentiment_score
```

After building the hybrid scorer, I defined another function to recommend the top N (5 in this case) products to the user and visualized the recommendations using plotly.

```ruby
def recommend_products(user_id, top_n=5):
    # Get all unique product IDs
    all_products = df['product_id'].unique()
    
    # Score all products
    scores = [(pid, hybrid_score(user_id, pid)) for pid in all_products]
    
    # Sort by score and return top N
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_products = scores[:top_n]
    
    # Return product details
    return df[df['product_id'].isin([pid for pid, _ in top_products])][['product_id',
                                   'Product_name', 'Brand', 'Category', 'Type']].drop_duplicates()
```
```ruby
def visualize_recommendations(user_id, top_n=5):
    # Score all products
    all_products = df['product_id'].unique()
    scores = [(pid, hybrid_score(user_id, pid)) for pid in all_products]
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    top_df = df[df['product_id'].isin([pid for pid, _ in top_scores])]
    top_df = top_df.drop_duplicates(subset=['product_id'])[['product_id', 'Product_name', 'Brand', 'sentiment_norm']]
    
    top_df['score'] = [score for _, score in top_scores]
    
    fig = px.bar(top_df, x='Product_name', y='score', color='sentiment_norm',
                 hover_data=['Brand'],
                 title=f"Top {top_n} Recommendations for Customer {user_id}",
                 labels={'score': 'Hybrid Score', 'sentiment_norm': 'Sentiment'})
    fig.show()
```
## Recommending Products to Users

```ruby
user_id = 'CUST09858'
visualize_recommendations(user_id, top_n=3)

user_id = 'CUST00034'
visualize_recommendations(user_id, top_n=3)
```
![alt text](/img/CUST00034.png "bar Plot")
![alt text](/img/CUST09858.png "bar Plot")

Two users with similar purchase histories received different top recommendations due to differing sentiment trends in the reviews of commonly bought items. Rather than relying on static sentiment labels, I used a deep learning approach to generate dynamic sentiment scores based on text tone and content. These scores were then used to influence product recommendations. By integrating sentiment analysis into a hybrid recommender, I have shown how to translate customer feedback into data-driven personalization strategies that directly support business KPIs like retention, satisfaction, and revenue.

