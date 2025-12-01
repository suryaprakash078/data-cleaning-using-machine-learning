# data-cleaning-using-machine-learning
Author: Surya Prakash K T (Roll No: 24BAI156)

📌 Project Overview
This repository contains two Python programs designed to demonstrate a complete data science pipeline, including:
Data Pre-processing & Visualization
Machine Learning Classification using TF-IDF + Naive Bayes
The project uses the CICIDS 2017 Network Intrusion Detection Dataset, which contains normal and attack traffic instances for intrusion and fraud detection research.

📁 Contents
/project-root
│── Program1_DataCleaning_Visualization.ipynb
│── Program2_ML_TFIDF_NaiveBayes.ipynb
│── cleaned_dataset.csv
│── dashboard.png
│── README.md
│── /dataset
│     └── CICIDS2017.csv  (not uploaded due to size)

🔍 Program 1: Data Pre-processing & Visualization
✔ Features

This program performs essential data-cleaning tasks:

Upload Excel dataset using Google Colab

Load dataset using pandas.read_excel()

View dataset structure using:

data.info()

df.describe()

df.isnull().sum()

Handle missing values:

Drop null values

Fill missing phone numbers with mean

Fill missing emails with "unknown"

Remove duplicates:

df.drop_duplicates(inplace=True)

Generate visualizations using Matplotlib:

Pie chart

Scatter plot

Bar chart

Combined dashboard using plt.subplot()

Save cleaned dataset:

df.to_csv("updated.csv", index=False)

Export dashboard as:

dashboard.png

📊 Program 2: Machine Learning Classification
✔ Dataset Used

CICIDS 2017 Dataset
A real-world network intrusion and fraud detection dataset containing:

Normal traffic

DDoS, DoS, Brute Force, Infiltration & botnet attacks

80+ traffic features

Labels: "Benign" vs various attack types

Used for binary and multi-class intrusion detection.

✔ ML Pipeline

This program builds a text-based ML model using:

1) Data Preparation

Upload dataset using files.upload()

Separate features (X) and labels (y)

Train-test split: test_size=0.2

2) Text Conversion

Since TF-IDF requires text, each row is converted to a single string:

X_train_text = X_train.astype(str).agg(' '.join, axis=1)
X_test_text = X_test.astype(str).agg(' '.join, axis=1)

3) TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

4) Naive Bayes Classification
model = MultinomialNB()
model.fit(X_train_vec, y_train)
predictions = model.predict(X_test_vec)

5) Evaluation Metrics

Accuracy Score

Confusion Matrix (with heatmap)

ROC Curve (if dataset has 2 classes)

📈 Visual Outputs
Program 1

Pie Chart

Scatter Plot

Bar Graph

Combined Dashboard (dashboard.png)

Cleaned CSV file (updated.csv)

Program 2

Class distribution bar graph

Confusion Matrix (heatmap)

ROC Curve (if binary classification)

⚙️ Installation & Requirements
Install Dependencies
pip install pandas numpy matplotlib scikit-learn

Run in Google Colab

Upload dataset when prompted.

▶️ How to Run the Programs
Program 1

Open Program1_DataCleaning_Visualization.ipynb

Upload Excel file

Run all cells

Check output:

updated.csv

dashboard.png

Program 2

Open Program2_ML_TFIDF_NaiveBayes.ipynb

Upload CICIDS dataset CSV

Run all cells

View:

Accuracy Score

Confusion Matrix

ROC Curve

📌 Conclusion

This repository demonstrates a complete end-to-end Data Science + Machine Learning workflow:

Dataset loading

Cleaning & preprocessing

Visualization

Text-based feature engineering

Naive Bayes classification

Performance evaluation

The project is a practical example of how Python can be applied to fraud detection and intrusion detection systems.

📞 Author

Name: Surya Prakash K T
Roll No: 24BAI156
