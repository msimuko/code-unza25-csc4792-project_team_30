import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Import the clean_reference function from src/utils.py
from src.utils import clean_reference

# 1. Load the data
DATA_PATH = 'data/references.csv'
df = pd.read_csv(DATA_PATH)

# 2. Preprocess the reference_text column
df['clean_text'] = df['reference_text'].apply(clean_reference)

# 3. Split the data into train and test sets (stratified)
X = df['clean_text']
y = df['publication_type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )),
    ('clf', MultinomialNB(alpha=1.0))
])

# 5. Train the model
pipeline.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = pipeline.predict(X_test)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Show top 20 TF-IDF features
tfidf = pipeline.named_steps['tfidf']
clf = pipeline.named_steps['clf']
if hasattr(clf, 'feature_log_prob_'):
    feature_names = tfidf.get_feature_names_out()
    for i, class_label in enumerate(clf.classes_):
        print(f"\nTop 20 features for class '{class_label}':")
        top20 = clf.feature_log_prob_[i].argsort()[::-1][:20]
        print([feature_names[j] for j in top20])

# 8. Save the trained pipeline
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'publication_classifier.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"\nTrained model saved to {MODEL_PATH}")
