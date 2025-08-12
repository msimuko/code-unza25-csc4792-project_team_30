import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from src.utils import clean_reference

DATA_PATH = "data/references.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "publication_classifier.pkl")
NOTEBOOK_PATH = "notebooks/classification_model.ipynb"

st.set_page_config(page_title="Reference Publication Type Classification", layout="wide")

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame(columns=["reference_text", "publication_type"])

def save_data(df):
    df.to_csv(DATA_PATH, index=False)

def append_data(reference_text, publication_type):
    df = load_data()
    new_row = {"reference_text": reference_text, "publication_type": publication_type}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df)

def delete_rows(indices):
    df = load_data()
    df = df.drop(indices).reset_index(drop=True)
    save_data(df)

def train_model(df):
    df['clean_text'] = df['reference_text'].apply(clean_reference)
    X = df['clean_text']
    y = df['publication_type']
    class_counts = y.value_counts()
    use_stratify = class_counts.min() >= 2 and len(class_counts) > 1
    if not use_stratify:
        stratify = None
    else:
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    labels = pipeline.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return acc, report, cm, pipeline, y_test, y_pred, not use_stratify

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ("View Data", "Train Model", "Classify Reference", "Ask Publication Type", "Notebooks")
    )

    if page == "View Data":
        st.header("📄 View and Manage Reference Data")
        df = load_data()
        st.dataframe(df, use_container_width=True)
        st.markdown("### Add New Reference")
        with st.form("add_reference_form"):
            ref_text = st.text_area("Reference Text")
            pub_type = st.text_input("Publication Type")
            submitted = st.form_submit_button("Add Reference")
            if submitted and ref_text.strip() and pub_type.strip():
                append_data(ref_text.strip(), pub_type.strip())
                st.success("Reference added.")
                st.experimental_rerun()
        st.markdown("### Delete Selected Rows")
        if not df.empty:
            selected = st.multiselect(
                "Select rows to delete (by index):",
                df.index.tolist()
            )
            if st.button("Delete Selected"):
                if selected:
                    delete_rows(selected)
                    st.success("Selected rows deleted.")
                    st.experimental_rerun()
                else:
                    st.warning("No rows selected.")

    elif page == "Train Model":
        st.header("🧑‍💻 Train Publication Type Classifier")
        df = load_data()
        if df.empty:
            st.warning("No data available. Please add references first.")
            return
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                acc, report, cm, pipeline, y_test, y_pred, non_stratified = train_model(df)
            st.success(f"Model trained! Accuracy: {acc:.2%}")
            if non_stratified:
                st.warning("Stratified split was not possible (some classes have <2 samples). Used random split instead.")
            st.markdown("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())
            st.markdown("#### Confusion Matrix")
            labels = pipeline.classes_
            st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))
            st.info(f"Trained model saved to `{MODEL_PATH}`.")
        else:
            st.info("Click 'Train Model' to start training.")

    elif page == "Classify Reference":
        st.header("🔎 Classify a Reference")
        model = load_model()
        if model is None:
            st.warning("No trained model found. Please train the model first.")
            return
        ref_input = st.text_area("Enter a reference to classify:")
        if st.button("Classify"):
            if not ref_input.strip():
                st.warning("Please enter a reference.")
            else:
                cleaned = clean_reference(ref_input)
                pred = model.predict([cleaned])[0]
                proba = model.predict_proba([cleaned])[0]
                conf = proba.max()
                st.success(f"Predicted Type: **{pred}** (Confidence: {conf:.2%})")
                st.markdown("##### Class Probabilities")
                prob_df = pd.DataFrame({
                    "Type": model.classes_,
                    "Probability": proba
                }).sort_values("Probability", ascending=False)
                st.dataframe(prob_df, use_container_width=True)

    elif page == "Ask Publication Type":
        st.header("❓ Ask: What Publication Type is this Reference?")
        model = load_model()
        if model is None:
            st.warning("No trained model found. Please train the model first.")
            return
        ref_input = st.text_area("Paste your reference below to ask its publication type:")
        if st.button("Get Publication Type"):
            if not ref_input.strip():
                st.warning("Please enter a reference.")
            else:
                cleaned = clean_reference(ref_input)
                pred = model.predict([cleaned])[0]
                proba = model.predict_proba([cleaned])[0]
                conf = proba.max()
                st.success(f"Publication Type: **{pred}** (Confidence: {conf:.2%})")
                st.markdown("##### Class Probabilities")
                prob_df = pd.DataFrame({
                    "Type": model.classes_,
                    "Probability": proba
                }).sort_values("Probability", ascending=False)
                st.dataframe(prob_df, use_container_width=True)

    elif page == "Notebooks":
        st.header("📓 Project Notebook")
        if os.path.exists(NOTEBOOK_PATH):
            st.markdown("You can download or open the main Jupyter notebook below:")
            with open(NOTEBOOK_PATH, "rb") as f:
                st.download_button(
                    label="Download classification_model.ipynb",
                    data=f,
                    file_name="classification_model.ipynb",
                    mime="application/x-ipynb+json"
                )
            st.markdown(
                f"[Open in Jupyter (if running locally)](file://{os.path.abspath(NOTEBOOK_PATH)})"
            )
        else:
            st.warning("Notebook not found.")

if __name__ == "__main__":
    main()
