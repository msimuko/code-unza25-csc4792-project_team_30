#!/usr/bin/env python3
"""
Reference Classification Model
CSC 4792 - University of Zambia (UNZA)

Complete machine learning pipeline to classify academic references by publication type.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def clean_reference(ref):
    """
    Clean and preprocess reference text for machine learning.
    
    Args:
        ref (str): Raw reference text
        
    Returns:
        str: Cleaned reference text
    """
    if not isinstance(ref, str):
        return ""
    
    # Convert to lowercase
    ref = ref.lower()
    
    # Strip leading/trailing whitespace
    ref = ref.strip()
    
    # Remove newline characters
    ref = ref.replace('\n', ' ').replace('\r', ' ')
    
    # Remove extra whitespace
    ref = re.sub(r'\s+', ' ', ref)
    
    # Keep only alphanumeric characters and basic punctuation
    # Allow: letters, numbers, spaces, periods, commas, colons, semicolons, hyphens, parentheses
    ref = re.sub(r'[^a-z0-9\s\.,;:\-\(\)]', '', ref)
    
    # Remove extra spaces again after punctuation removal
    ref = re.sub(r'\s+', ' ', ref).strip()
    
    return ref

def create_sample_dataset():
    """
    Create a sample dataset with realistic academic references.
    
    Returns:
        pandas.DataFrame: Dataset with reference_text and publication_type columns
    """
    
    references_data = [
        # Journal Articles
        ("Smith, J.A., Johnson, M.B. (2023). Machine Learning Applications in Academic Research. Journal of Computer Science, 45(3), 123-145.", "Journal Article"),
        ("Brown, K.L., Davis, R.T., Wilson, P.Q. (2022). Natural Language Processing for Text Classification. IEEE Transactions on AI, 15(8), 234-256.", "Journal Article"),
        ("Garcia, M.S., Lee, C.H. (2024). Deep Learning Models for Reference Analysis. Nature Machine Intelligence, 7(2), 89-102.", "Journal Article"),
        ("Anderson, D.P., Thompson, L.M., White, J.K. (2023). Statistical Methods in Information Retrieval. ACM Computing Surveys, 55(4), 1-28.", "Journal Article"),
        ("Martinez, R.A., Kumar, S. (2022). Text Mining Techniques for Academic Literature. Information Processing & Management, 58(6), 102-118.", "Journal Article"),
        ("Taylor, B.N., Clark, F.S., Miller, H.R. (2024). Automated Classification Systems in Libraries. Library & Information Science Research, 46(1), 45-62.", "Journal Article"),
        
        # Books
        ("Johnson, A.B. (2023). Introduction to Machine Learning. 3rd Edition. MIT Press, Cambridge, MA.", "Book"),
        ("Williams, C.D., Brown, E.F. (2022). Natural Language Processing: Theory and Practice. Springer-Verlag, New York.", "Book"),
        ("Davis, G.H. (2024). Data Science Fundamentals. Oxford University Press, Oxford, UK.", "Book"),
        ("Thompson, K.L., Anderson, M.P., Wilson, R.Q. (2023). Artificial Intelligence in Education. Pearson Education, Boston.", "Book"),
        ("Rodriguez, S.A. (2022). Python for Data Analysis: Advanced Techniques. O'Reilly Media, Sebastopol, CA.", "Book"),
        ("Chen, L.W., Patel, N.K. (2024). Deep Learning with TensorFlow. Manning Publications, Shelter Island, NY.", "Book"),
        
        # Conference Papers
        ("Lee, H.J., Kim, S.M. (2023). Automated Reference Classification Using NLP. Proceedings of the 41st International Conference on Machine Learning, pp. 1234-1245.", "Conference Paper"),
        ("Patel, R.K., Sharma, A.N. (2022). Text Classification with Transformer Models. IEEE International Conference on Data Mining, pp. 567-578.", "Conference Paper"),
        ("Wang, X.L., Zhang, Y.H. (2024). Feature Engineering for Academic Text Analysis. Annual Conference of the Association for Computational Linguistics, pp. 890-902.", "Conference Paper"),
        ("Jackson, P.T., Moore, L.S. (2023). Comparative Study of Classification Algorithms. International Conference on Artificial Intelligence, pp. 345-356.", "Conference Paper"),
        ("Kumar, V.P., Gupta, R.M. (2022). Ensemble Methods for Text Classification. Conference on Neural Information Processing Systems, pp. 2134-2147.", "Conference Paper"),
        
        # Thesis/Dissertation
        ("Mitchell, S.L. (2023). Machine Learning Approaches to Academic Reference Classification. PhD Dissertation, Stanford University, Department of Computer Science.", "Thesis"),
        ("Roberts, J.M. (2022). Natural Language Processing for Bibliographic Data Analysis. Master's Thesis, University of California Berkeley, School of Information.", "Thesis"),
        ("Turner, K.R. (2024). Automated Text Classification Systems in Digital Libraries. PhD Dissertation, Carnegie Mellon University, Computer Science Department.", "Thesis"),
        ("Adams, P.C. (2023). Deep Learning Models for Citation Analysis. Master's Thesis, Georgia Institute of Technology, College of Computing.", "Thesis"),
        
        # Reports
        ("National Science Foundation. (2023). Trends in Academic Publishing and Digital Libraries. Technical Report NSF-23-145, Washington, DC.", "Report"),
        ("World Health Organization. (2022). Guidelines for Systematic Literature Reviews in Health Sciences. WHO Technical Report Series, No. 975, Geneva.", "Report"),
        ("U.S. Department of Education. (2024). Educational Technology Integration: Current Status and Future Directions. Report ED-2024-032, Washington, DC.", "Report"),
        ("European Commission. (2023). Digital Transformation in Higher Education Institutions. Research Report EC-2023-089, Brussels, Belgium.", "Report"),
        
        # Web Resources
        ("Python Software Foundation. (2024). Scikit-learn: Machine Learning in Python. Retrieved from https://scikit-learn.org/stable/", "Web Resource"),
        ("Google AI. (2023). TensorFlow Documentation and Tutorials. Available at: https://www.tensorflow.org/", "Web Resource"),
        ("Stack Overflow Community. (2024). Best Practices for Text Classification in Python. Retrieved from https://stackoverflow.com/questions/text-classification", "Web Resource"),
        ("GitHub Inc. (2023). Natural Language Toolkit (NLTK) Documentation. Available at: https://github.com/nltk/nltk", "Web Resource"),
        ("Towards Data Science. (2024). Advanced Techniques in Machine Learning for NLP. Medium Platform. Retrieved from https://towardsdatascience.com/", "Web Resource"),
        
        # Additional diverse examples
        ("OpenAI. (2023). GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.", "Report"),
        ("Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30, 5998-6008.", "Conference Paper")
    ]
    
    df = pd.DataFrame(references_data, columns=['reference_text', 'publication_type'])
    return df

def build_and_evaluate_model(df):
    """
    Build, train, and evaluate the classification model.
    
    Args:
        df (pandas.DataFrame): Dataset with references and labels
    """
    
    print("=" * 60)
    print("REFERENCE CLASSIFICATION MODEL")
    print("CSC 4792 - University of Zambia (UNZA)")
    print("=" * 60)
    
    # Data preprocessing
    print(f"\n📊 Dataset Information:")
    print(f"Total references: {len(df)}")
    print(f"Publication types: {df['publication_type'].nunique()}")
    print(f"\nDistribution of publication types:")
    print(df['publication_type'].value_counts().to_string())
    
    # Clean the reference texts
    print(f"\n🧹 Cleaning reference texts...")
    df['cleaned_reference'] = df['reference_text'].apply(clean_reference)
    
    # Prepare features and labels
    X = df['cleaned_reference']
    y = df['publication_type']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build classification pipelines
    models = {
        'Multinomial Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=1.0))
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    
    # Train and evaluate both models
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, pipeline in models.items():
        print(f"\n🤖 Training {name}...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = pipeline
            best_name = name
        
        # Print classification report
        print(f"\n📈 Classification Report - {name}:")
        print("-" * 50)
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print(f"\n📊 Confusion Matrix - {name}:")
        print("-" * 40)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    
    print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_score:.4f})")
    
    # Feature analysis for best model
    print(f"\n🔍 Top Features Analysis ({best_name}):")
    print("-" * 45)
    
    feature_names = best_model.named_steps['tfidf'].get_feature_names_out()
    
    if hasattr(best_model.named_steps['classifier'], 'feature_log_prob_'):
        # For Naive Bayes
        feature_probs = best_model.named_steps['classifier'].feature_log_prob_
        classes = best_model.named_steps['classifier'].classes_
        
        for i, class_name in enumerate(classes):
            top_features_idx = feature_probs[i].argsort()[-5:][::-1]
            top_features = [feature_names[idx] for idx in top_features_idx]
            print(f"{class_name}: {', '.join(top_features)}")
    
    elif hasattr(best_model.named_steps['classifier'], 'coef_'):
        # For Logistic Regression
        if len(best_model.classes_) == 2:
            coef = best_model.named_steps['classifier'].coef_[0]
            top_pos_idx = coef.argsort()[-5:][::-1]
            top_neg_idx = coef.argsort()[:5]
            
            print(f"Most positive features: {', '.join([feature_names[i] for i in top_pos_idx])}")
            print(f"Most negative features: {', '.join([feature_names[i] for i in top_neg_idx])}")
        else:
            # Multi-class
            classes = best_model.named_steps['classifier'].classes_
            for i, class_name in enumerate(classes):
                coef = best_model.named_steps['classifier'].coef_[i]
                top_idx = coef.argsort()[-3:][::-1]
                top_features = [feature_names[idx] for idx in top_idx]
                print(f"{class_name}: {', '.join(top_features)}")
    
    # Demonstrate predictions on new examples
    print(f"\n🎯 Sample Predictions:")
    print("-" * 30)
    
    sample_refs = [
        "Johnson, A. (2024). Deep Learning with Python. Manning Publications.",
        "Smith, B. et al. (2023). AI in Healthcare. Journal of Medical Informatics, 15(3), 45-67.",
        "Brown, C. (2024). Machine Learning Applications. PhD Thesis, MIT.",
        "AI Conference Proceedings. (2023). Neural Networks in Practice. IEEE, pp. 123-145."
    ]
    
    for ref in sample_refs:
        cleaned_ref = clean_reference(ref)
        prediction = best_model.predict([cleaned_ref])[0]
        confidence = max(best_model.predict_proba([cleaned_ref])[0])
        
        print(f"Reference: {ref[:60]}...")
        print(f"Predicted Type: {prediction} (Confidence: {confidence:.3f})")
        print()
    
    return best_model

def main():
    """
    Main function to run the complete classification pipeline.
    """
    
    # Create sample dataset
    print("🔄 Creating sample dataset...")
    df = create_sample_dataset()
    
    # Build and evaluate model
    model = build_and_evaluate_model(df)
    
    print("\n✅ Model training and evaluation completed!")
    print("The classification pipeline is ready for use with real data.")
    print("\nTo use with your own data:")
    print("1. Replace the sample dataset with your references.csv")
    print("2. Ensure columns are named 'reference_text' and 'publication_type'")
    print("3. Run this script to train and evaluate the model")
    
    return model

if __name__ == "__main__":
    # Run the complete pipeline
    trained_model = main()