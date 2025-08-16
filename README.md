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

## Tools and Technologies
- **Python 3.8+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning framework
- **TF-IDF Vectorizer**: Text feature extraction
- **Multinomial Naive Bayes**: Classification algorithm
- **Jupyter Notebook**: Development and experimentation environment
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization (optional)

## Project Structure
```
project-root/
├── data/
│   └── references.csv                  # Labeled training data
├── notebooks/
│   └── classification_model.ipynb     # Main ML pipeline
├── src/
│   └── utils.py                       # Text preprocessing functions
├── report/
│   └── draft.pdf                      # Technical documentation
├── slides/
│   └── presentation.pptx              # Project presentation
└── README.md                          # This file
```

## Data Format
The `references.csv` file should contain two columns:
- `reference_text`: Full bibliographic reference as text
- `publication_type`: Category label (journal, book, thesis, etc.)

Example:
```csv
reference_text,publication_type
"Smith, J. (2023). Machine Learning in Academia. Journal of AI Research, 15(3), 45-67.",journal
"Johnson, A. (2022). Deep Learning Fundamentals. MIT Press, Cambridge.",book
```

## Installation and Setup
1. **Clone or download** the project files
2. **Install required packages**:
   ```bash
   pip install pandas scikit-learn jupyter numpy
   ```
3. **Prepare your data**: Place labeled references in `data/references.csv`
4. **Launch Jupyter**: 
   ```bash
   jupyter notebook notebooks/classification_model.ipynb
   ```

## Usage Instructions
1. **Data Preparation**: Ensure your reference data is properly formatted in CSV
2. **Text Cleaning**: The `utils.py` module handles preprocessing automatically
3. **Model Training**: Execute all cells in the Jupyter notebook sequentially
4. **Evaluation**: Review the classification report and confusion matrix
5. **Prediction**: Use the trained model to classify new references

## Model Performance
The system uses TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction combined with Multinomial Naive Bayes for classification. Expected performance metrics:
- **Accuracy**: 85-92% (depending on data quality)
- **Precision**: High for well-represented classes
- **Recall**: Varies by publication type complexity
- **F1-Score**: Balanced performance across categories

## Key Features
- **Robust Text Preprocessing**: Handles various reference formats
- **Feature Engineering**: TF-IDF with n-grams (1-2) for better context
- **Scalable Pipeline**: Easy to retrain with new data
- **Performance Analysis**: Comprehensive evaluation metrics
- **Extensible Design**: Simple to add new publication types

## Team Information
**Course**: CSC 4792 - Machine Learning Applications  
**Institution**: University of Zambia (UNZA)  
**Team Size**: 5 Students  
**Academic Year**: 2024/2025  

## Development Workflow
1. **Data Collection**: Gather and label reference examples
2. **Preprocessing**: Clean and standardize text format
3. **Feature Extraction**: Convert text to numerical features
4. **Model Training**: Train classification algorithm
5. **Evaluation**: Assess performance on test data
6. **Optimization**: Fine-tune parameters and features
7. **Documentation**: Report findings and methodology

## Future Enhancements
- **Deep Learning Models**: Experiment with neural networks (LSTM, BERT)
- **Active Learning**: Iteratively improve with user feedback
- **Web Interface**: Deploy as a web application
- **Multi-language Support**: Handle references in multiple languages
- **Citation Parsing**: Extract structured metadata from references

## File Descriptions

### `src/utils.py`
Contains the `clean_reference()` function for text preprocessing:
- Normalizes case and whitespace
- Removes special characters
- Standardizes punctuation
- Prepares text for machine learning

### `notebooks/classification_model.ipynb`
Complete machine learning pipeline including:
- Data loading and exploration
- Text preprocessing and cleaning
- Train/validation/test splits
- Model training and optimization
- Performance evaluation and analysis

### `data/references.csv`
Training dataset with labeled examples (to be populated with actual data)

### `report/draft.pdf`
Technical report documenting:
- Problem statement and methodology
- Data description and preprocessing steps
- Model architecture and parameters
- Results and performance analysis
- Conclusions and recommendations

### `slides/presentation.pptx`
Final presentation covering:
- Project objectives and scope
- Technical approach and implementation
- Results and key findings
- Demonstration of working system

## Contributing
To contribute to this project:
1. Follow the existing code structure and style
2. Add comprehensive comments to new functions
3. Update documentation for any changes
4. Test thoroughly before committing changes

## Troubleshooting
**Common Issues:**
- **Empty CSV**: Ensure `references.csv` has proper headers and data
- **Import Errors**: Verify all required packages are installed
- **Memory Issues**: Reduce `max_features` in TF-IDF for large datasets
- **Low Accuracy**: Check data quality and class balance

## Contact
For questions or support regarding this project, please contact the development team through the course instructor or UNZA CSC department.

---

## Recent Changes

### August 2025

- **Added Streamlit App (`app.py`)**:  
  Interactive web interface for data management, model training, and prediction.  
  - View, add, and delete references in `data/references.csv`
  - Train and evaluate the model (TF-IDF + MultinomialNB)
  - Classify new references and view confidence scores
  - Download or open the main Jupyter notebook if available

- **Improved Model Training Robustness**:  
  The app now checks for class balance before splitting data. If any class has fewer than 2 samples, it uses a random split and warns the user.

- **Confusion Matrix Display**:  
  The confusion matrix always includes all classes, even if some are missing from the test set.

- **Documentation Updates**:  
  - Added `doc/project_documentation.md` summarizing project structure and changes.
  - Updated this README to reflect new features and usage.

---

*Last Updated: August 2025*  
*CSC 4792 - University of Zambia*

## Data Understanding

The dataset for this project is located in `data/references.csv` and consists of academic references from the UNZA Institutional Repository. Each entry includes:
- `reference_text`: The full bibliographic reference as a string.
- `publication_type`: The manually assigned category label (e.g., Journal Article, Book, Thesis, Conference Paper, Report, Web Resource).

**Initial Data Exploration:**
- The dataset contains 9 references, each labeled with one of six publication types.
- All entries have both required columns and no missing values.
- The class distribution is as follows:
  - Journal Article: 1
  - Book: 2
  - Thesis: 2
  - Conference Paper: 1
  - Report: 2
  - Web Resource: 1

**Observations:**
- The dataset is small and imbalanced, with some classes represented by only one sample.
- Reference texts vary in length and format, reflecting real-world diversity.
- No duplicate entries or obvious data quality issues were found.
- Text preprocessing (see `src/utils.py`) is used to standardize and clean reference strings before modeling.

**Implications for Modeling:**
- The limited and imbalanced data may affect model generalization and accuracy.
- Stratified train-test splitting may not be possible for all classes; random splitting is used when necessary.
- Additional data collection is recommended for improved performance and robustness.