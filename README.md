# 🎬 IMDB Movies Review

This project analyzes IMDB movie reviews to extract meaningful insights using various natural language processing techniques. The primary goal is to process the review data, apply text preprocessing, and build a sentiment analysis model using machine learning algorithms.

## 📝 Overview

- This project leverages the power of machine learning to analyze and classify movie reviews from IMDB.
- The dataset consists of movie reviews from IMDB with sentiment labels.
- The goal is to classify reviews into positive and negative sentiment categories.

## 🚀 Features

- Data preprocessing and cleaning
- Word embedding techniques (e.g., tokenizer)
- Sentiment analysis using machine learning algorithms
- Visualization of sentiment distribution

## 📁 Project Structure
```
.
├── Data
│   ├── IMDB_Dataset.csv
│   ├── word_index.json
│   ├── X_encoded.npy
│   ├── X_test.npy
│   ├── X_train.npy
│   ├── y_encoded.npy
│   ├── y_test.npy
│   └── y_train.npy
├── Encoder
│   └── tokenizer.pickle
├── Model
│   ├── Model.h5
│   └── Model_scratch.h5
├── Notebook
│   ├── Model_Training.ipynb
│   ├── Model_Training_scratch.ipynb
│   ├── prediction.ipynb
│   ├── prediction_scratch.ipynb
│   └── Word_Embedding.ipynb
├── prediction
│   ├── prediction.py
│   └── prediction_scratch.py
├── README.md
├── app.py
├── app_scratch_model.py
└── requirements.txt
```

## 🔧 Technologies Used

- **Python**: Main programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
- **TensorFlow / Keras**: Deep learning models
- **NLTK / SpaCy**: Natural language processing tools


#
## 🛠️ Installation

**1. Clone the repository:**
```
   git clone https://github.com/Raj-UtsaV/Bank-Customer-Churn-Predictor.git
   cd Bank-Customer-Churn-Predictor
```

**2. Create and activate a virtual environment (optional but recommended):**
```
   python -m venv venv
```
*On Windows*
```
venv\Scripts\activate
```
*On Unix or MacOS*
```
source venv/bin/activate
```

**3. Install the required packages:**
```
   pip install -r requirements.txt
```

### 📊 Data Description
The dataset used in this project consists of IMDB movie reviews along with their sentiment labels. Each review is labeled as either positive or negative. The data is used for training and evaluating the sentiment classification model.

## 🚀 Usage
You can train the model ,
Once the model is trained, you can use it to predict the sentiment of new movie reviews. You can also visualize the distribution of sentiments and generate insights based on the review text.

**OR**

*run the pretrained model webapp*
```
python app.py
```





#
## 📬 Contact
- Name: Utsav Raj
- LinkedIn: https://www.linkedin.com/in/utsav-raj-6657b12bb
- Email: utsavraj911@outlook.com
