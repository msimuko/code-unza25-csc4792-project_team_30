# Reference Classification Project

## Overview
This project develops an automated system to classify academic references by publication type using machine learning techniques. The system can distinguish between journals, books, theses, conference papers, and other publication formats, helping researchers and librarians organize academic literature more efficiently.

## Objective
To build and evaluate a text classification model that accurately categorizes bibliographic references into different publication types based on their textual content and formatting patterns.

## Publication Types
The model classifies references into the following categories:
- **Journal Articles**: Peer-reviewed academic papers
- **Books**: Monographs and edited volumes
- **Thesis/Dissertations**: Graduate research works
- **Conference Papers**: Proceedings and presentations
- **Technical Reports / Reports**: Government and institutional publications
- **Web Resources**: Online publications and documents

> **Note:** The project and app support both "Report" and "Technical Report" as valid types.

# 📘 UNZA CSC4792 Project — Team 30

Machine Learning pipeline for classifying academic references into publication types.  

This project runs fully in **Google Colab** and supports:
- **Demo Mode** (use preloaded dataset rows for testing).  
- **Deployment Mode** (paste bitstream URLs for live classification).  

---

## ⚙️ Requirements
- Google Colab (recommended) or Jupyter Notebook  
- Python ≥ 3.9  
- Dependencies (already handled inside the notebook):  
  ```bash
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  joblib
  requests
  python-docx
  PyPDF2
📂 Project Structure
bash
Copy code
/project_root
│── project_notebook.ipynb     # Main Colab notebook
│── /data
│     └── model_evaluation_summary.csv
│── /models
│     ├── svm_model.pkl
│     └── vectorizer.pkl
│── /outputs
│     └── references_cleaned.csv
▶️ Quick Start (Colab)
Open the notebook
Upload or open project_notebook.ipynb in Google Colab.

Mount Google Drive
The notebook will prompt you to mount your Google Drive. Ensure this path exists:


🚀 Deployment Usage of model 
After training, the Deployment Stage enables classification in two modes:

Option 1 — Demo Mode
Run with built-in dataset rows to check predictions:

python
Copy code
predict_from_demo(sample internal url provided for testing )
Option 2 — Manual Input (Bitstream URL)
Paste a PDF or DOCX bitstream URL from UNZA DSpace (or similar).

python
Copy code
url = "https://dspace.unza.zm/bitstream/handle/123456789/example.pdf"
predict_from_url(url)
Output Example:

makefile
Copy code
Prediction: Thesis
Confidence: 0.91
🧪 Testing Checklist
 Run all notebook cells without error

 Verify cleaned dataset is saved in /outputs

 Confirm evaluation summary exists in /data

 Simply Paste a bitstream URL in the input field  and get a classification result

👥 Team 30
