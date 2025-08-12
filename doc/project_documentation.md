# Reference Publication Type Classification Project Documentation

## Project Overview

This project aims to automate the classification of academic references by publication type using machine learning. The system distinguishes between journals, books, theses, conference papers, reports, and web resources, enabling efficient organization of academic literature.

## Directory Structure

```
project-root/
├── data/
│   └── references.csv
├── doc/
│   └── project_documentation.md
├── model/
│   └── publication_classifier.pkl
├── notebooks/
│   └── classification_model.ipynb
├── report/
│   └── draft.pdf
├── slides/
│   └── presentation.pptx
├── src/
│   ├── classification_model.py
│   └── utils.py
├── train_model.py
└── README.md
```

## Key Components

- **src/utils.py**: Contains `clean_reference()` for text preprocessing.
- **src/classification_model.py**: Sample pipeline for model training and evaluation using synthetic data.
- **train_model.py**: Script for training, evaluating, and saving the classifier using real data from `data/references.csv`.
- **model/publication_classifier.pkl**: Saved trained scikit-learn pipeline.
- **README.md**: Project overview, setup, and usage instructions.

## Summary of Changes

### 1. Added `train_model.py`
- Loads `data/references.csv`.
- Uses `clean_reference()` from `src/utils.py` for preprocessing.
- Splits data (stratified), builds a pipeline (TF-IDF + MultinomialNB), trains, evaluates, prints metrics, and saves the model to `model/publication_classifier.pkl`.
- Creates `model/` directory if missing.
- Well-commented and ready to run from project root.

### 2. Documentation (this file)
- Summarizes project structure, workflow, and all code changes.
- Provides a reference for contributors and users.

## How to Use

1. Place your labeled data in `data/references.csv` with columns: `reference_text`, `publication_type`.
2. Run `python train_model.py` from the project root to train and evaluate the model.
3. The trained model will be saved in `model/publication_classifier.pkl`.

## Contribution Guidelines

- Follow code structure and style.
- Add comments to new functions.
- Update documentation for any changes.
- Test thoroughly before committing.

## Change Log

- **2024-2025**: Initial project setup, sample pipeline, and documentation.
- **August 2025**: Added `train_model.py` and `doc/project_documentation.md` for reproducible training and documentation.

---

*For further details, see the README.md and inline code comments.*
