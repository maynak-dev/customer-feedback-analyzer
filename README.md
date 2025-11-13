# 🧠 Customer Feedback Analyzer – Python NLP Project

A **Natural Language Processing (NLP)** project built using **Python, Pandas, NLTK (VADER), Matplotlib, and WordCloud** to analyze and visualize customer sentiments from text feedback.  
This tool automatically processes feedback data, performs sentiment classification (Positive, Negative, Neutral), and generates an easy-to-read summary report.

---

## 🚀 Overview

Businesses often receive large volumes of feedback daily.  
This project automates **sentiment detection and visualization** from customer feedback using **VADER Sentiment Analyzer**.

### 🎯 Key Features
- Clean and preprocess textual data  
- Perform **sentiment analysis** (Positive, Negative, Neutral)  
- Generate **word clouds** and **summary reports**  
- Export results to `.csv` and `.txt` for easy sharing  

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Libraries** | Pandas, Numpy, NLTK, Matplotlib, Seaborn, WordCloud |
| **Model** | VADER (Valence Aware Dictionary and sEntiment Reasoner) |
| **Environment** | Google Colab / Jupyter Notebook |
| **Output Files** | CSV (Results), TXT (Summary), PNG (Visuals) |

---

## 📂 Folder Structure

```
customer-feedback-analyzer/
│
├── customer_feedback_analyzer.py     # Main analysis script
├── Customer_Feedback_Analyzer.ipynb  # Notebook version
├── customer_feedback_results.csv     # Output with classified sentiments
├── Customer_Feedback_Report.txt      # Generated sentiment summary
│
├── requirements.txt                  # Dependencies list
└── README.md                         # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/maynak-dev/customer-feedback-analyzer.git
cd customer-feedback-analyzer
```

### 2️⃣ Install Dependencies
```bash
pip install pandas matplotlib seaborn nltk wordcloud
```

### 3️⃣ Download Required NLTK Resources
```python
import nltk
nltk.download('vader_lexicon')
```

---

## 🧠 How It Works

1. **Data Loading**  
   Reads feedback data from a CSV file and cleans any unusual quotes or extra symbols.

2. **Preprocessing**
   - Converts all text to lowercase  
   - Removes unwanted punctuation  
   - Handles missing data  

3. **Sentiment Analysis**
   - Uses **VADER** from `nltk.sentiment.vader`  
   - Calculates compound sentiment scores  
   - Categorizes as:
     - **Positive:** > 0.05  
     - **Negative:** < -0.05  
     - **Neutral:** otherwise  

4. **Visualization**
   - Word cloud of frequently used words  
   - Bar chart of sentiment distribution (optional)

5. **Report Generation**
   - Creates:
     - `customer_feedback_results.csv` → sentiment results  
     - `Customer_Feedback_Report.txt` → overall sentiment summary  

---

## 🧾 Example Output

### 🖥️ Console Summary
```
Customer Feedback Analysis Report
----------------------------------
Total Feedbacks: 96
Positive: 43
Negative: 33
Neutral: 20

Overall Sentiment: Positive
```

### 📊 Example Visualization
*(Word cloud generated from cleaned feedback text)*  
![Word Cloud](https://raw.githubusercontent.com/maynak-dev/customer-feedback-analyzer/main/wordcloud.png)

---

## 🧪 Code Example

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Load cleaned feedback
df = pd.read_csv("customer_feedback_results.csv")

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Apply sentiment scoring
df['Sentiment_Score'] = df['Text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['Sentiment'] = df['Sentiment_Score'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
)

# Save results
df.to_csv("customer_feedback_results.csv", index=False)
```

---

## 📈 Future Improvements
- Integration with live data sources (Twitter, product reviews API)
- Use advanced transformer models (BERT, RoBERTa)
- Build real-time dashboard using **Streamlit** or **Plotly Dash**
- Include emotion classification (joy, anger, sadness, etc.)

---

## 👨‍💻 Author
**Maynak Dey**  
📧 work.maynak@gmail.com  
🔗 [GitHub](https://github.com/maynak-dev) | [LinkedIn](https://www.linkedin.com/in/maynak-dey)

---

## 📝 License
Licensed under the **MIT License** — feel free to use, modify, and share.

---

## 🌟 Acknowledgements
- [NLTK Documentation](https://www.nltk.org/)
- [VADER Sentiment Analysis Paper](https://github.com/cjhutto/vaderSentiment)
- Python, Open Source, and Data Science Community
