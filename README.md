# Data Mining

A collection of **Data Mining and Machine Learning projects** implemented in Python. The repository covers unsupervised learning, time-series regression, and text classification using real-world datasets.

## Projects

### 1. COVID-19 Data Analysis & DBSCAN Clustering

Located in `Project 2022-2023/Ερώτημα 2/`.

This project processes COVID-19 data for multiple countries and uses **DBSCAN clustering** to explore relationships between epidemiological and socio-economic/environmental features.

Key steps:
- Clean and aggregate country-level COVID-19 data
- Calculate total tests, case positivity percentage, and death percentage
- Standardize numerical features using `StandardScaler`
- Apply `DBSCAN` clustering
- Visualize clusters interactively with Plotly
- Explore relationships such as positivity rate vs. average temperature, GDP per capita, healthcare capacity, and demographics

**Technologies:** Python, Pandas, Scikit-learn, Plotly

### 2. COVID-19 Cases Prediction with SVM

Located in `Project 2022-2023/Ερώτημα 3 - SVM/`.

This project focuses on predicting future COVID-19 cases in **Greece** using a Support Vector Regression (SVR) model.

Key steps:
- Extract and preprocess Greece-specific COVID-19 data
- Calculate daily cases and positivity rate
- Standardize the case data
- Create a time-based training/testing split
- Predict future cases using **Support Vector Regression** with a linear kernel
- Evaluate predictions using MAE, MSE, RMSE, and R²
- Visualize actual vs. predicted values

**Technologies:** Python, NumPy, Pandas, Matplotlib, Scikit-learn

### 3. Amazon Review Classification with Word2Vec & Random Forest

Located in `word2vec and random forest classification/`.

This project performs **text classification of Amazon reviews** using Word2Vec embeddings and a Random Forest classifier.

Key steps:
- Clean and tokenize review text
- Remove non-alphabetic characters and English stopwords
- Train a 100-dimensional Word2Vec model
- Represent each review using the average of its word vectors
- Split the dataset into training and testing sets
- Train a Random Forest classifier with 1,000 trees
- Evaluate the classifier using macro-averaged precision, recall, and F1-score

**Technologies:** Python, NumPy, Pandas, Gensim, NLTK, Scikit-learn

## Repository Structure

```text
Data-Mining/
├── Project 2022-2023/
│   ├── data.csv
│   ├── data_mining_project_2023.pdf
│   ├── Ερώτημα 2/
│   │   ├── Data_Mining.py
│   │   └── cleaned_data.csv
│   └── Ερώτημα 3 - SVM/
│       ├── SVM.py
│       └── cleaned_data.csv
│
└── word2vec and random forest classification/
    ├── main.py
    └── amazon.csv
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Tsomaros/Data-Mining.git
cd Data-Mining
```

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn plotly gensim nltk
```

For the text-classification project, download the NLTK English stopwords corpus:

```python
import nltk
nltk.download('stopwords')
```

## Running the Projects

Each project can be run independently from its respective directory.

### DBSCAN Clustering
```bash
cd "Project 2022-2023/Ερώτημα 2"
python Data_Mining.py
```

### SVM Regression
```bash
cd "Project 2022-2023/Ερώτημα 3 - SVM"
python SVM.py
```

### Word2Vec + Random Forest
```bash
cd "word2vec and random forest classification"
python main.py
```

## Methods & Concepts

- Data preprocessing and cleaning
- Feature engineering
- Data standardization
- DBSCAN clustering
- Support Vector Regression (SVR)
- Word2Vec word embeddings
- Random Forest classification
- Time-series train/test splitting
- Model evaluation
- Data visualization

## Evaluation Metrics

The projects use several metrics depending on the task:

| Task | Metrics |
|---|---|
| SVR regression | MAE, MSE, RMSE, R² |
| Text classification | Precision, Recall, F1-score |
| Clustering | DBSCAN cluster assignments and visualization |

## Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Plotly
- Gensim
- NLTK

## Documentation

The repository also contains the original project report in `Project 2022-2023/data_mining_project_2023.pdf`.

## License

This repository does not currently specify a license.