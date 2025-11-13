# 🧠 Customer Feedback Analyzer – Python & NLP

A **Natural Language Processing (NLP)** project built using **Python, Pandas, NLTK, and Matplotlib** to analyze and visualize customer sentiment from textual feedback.  
The tool automates text preprocessing, performs sentiment classification (Positive, Negative, Neutral), and generates easy-to-understand insights for management reporting.

---

## 🚀 Project Overview

Businesses receive thousands of customer reviews and feedback entries daily.  
This project helps summarize **customer emotions and satisfaction trends** automatically using **sentiment analysis** techniques.

### 🎯 Key Objectives
- Clean and preprocess raw text data (remove stopwords, punctuation, special characters)
- Perform sentiment analysis using `VADER` from the NLTK library
- Generate statistical summaries and sentiment distribution plots
- Automate feedback reporting for management decision-making

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Libraries** | Pandas, Numpy, NLTK, Matplotlib, Seaborn |
| **NLP Model** | VADER (Valence Aware Dictionary and sEntiment Reasoner) |
| **IDE / Tools** | VS Code, Jupyter Notebook |
| **Output** | Sentiment Summary CSV, Visualization Charts |

---

## 📂 Project Structure

```
customer-feedback-analyzer/
│
├── data/
│   ├── raw_feedback.csv          # Original feedback dataset
│   └── cleaned_feedback.csv      # After preprocessing
│
├── notebooks/
│   └── sentiment_analysis.ipynb  # Main notebook for analysis
│
├── reports/
│   ├── sentiment_distribution.png # Visualization of results
│   └── summary_report.txt         # Auto-generated report
│
├── src/
│   ├── data_preprocessing.py      # Text cleaning and normalization
│   ├── sentiment_model.py         # Sentiment scoring logic
│   └── visualization.py           # Graph generation
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/maynak-dev/customer-feedback-analyzer.git
cd customer-feedback-analyzer
```

### 2️⃣ Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK resources
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

---

## 🧠 How It Works

1. **Data Preprocessing**
   - Converts text to lowercase
   - Removes punctuation, special symbols, numbers, and stopwords
   - Tokenizes and lemmatizes words

2. **Sentiment Scoring**
   - Uses **VADER Sentiment Analyzer** from NLTK
   - Generates polarity scores (`positive`, `negative`, `neutral`, `compound`)
   - Classifies sentiment as:
     - **Positive:** compound ≥ 0.05  
     - **Negative:** compound ≤ -0.05  
     - **Neutral:** otherwise

3. **Visualization**
   - Bar chart showing percentage distribution of sentiments
   - Word cloud (optional)
   - Pie chart for sentiment breakdown

4. **Report Generation**
   - Summarizes:
     - Total reviews analyzed
     - Percentage of each sentiment type
     - Average sentiment score
   - Exports summary as a `.txt` report in `/reports`

---

## 📊 Example Output

### 🧾 Console Summary
```
Total Reviews Analyzed: 2500
Positive: 58.4%
Neutral: 28.1%
Negative: 13.5%
Average Sentiment Score: 0.23
```

### 📈 Visualization
![Sentiment Distribution](reports/sentiment_distribution.png)

---

## 🧪 Sample Code Snippet

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Load data
data = pd.read_csv("data/cleaned_feedback.csv")

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# Apply sentiment scoring
data["compound"] = data["Feedback"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
data["Sentiment"] = data["compound"].apply(
    lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
)

# Save results
data.to_csv("data/sentiment_results.csv", index=False)
```

---

## 📈 Future Enhancements
- Integrate with live APIs (Twitter, Google Reviews, etc.)
- Use transformer models like BERT for deeper sentiment understanding
- Build a dashboard with Plotly or Streamlit for real-time analysis
- Add topic modeling for common themes in feedback

---

## 👨‍💻 Author
**Maynak Dey**  
[🔗 GitHub](https://github.com/maynak-dev) | [🔗 LinkedIn](https://www.linkedin.com/in/maynak-dey)  
📧 work.maynak@gmail.com

---

## 📝 License
This project is licensed under the **MIT License** — feel free to use and modify it for your own work.

---

## 🌟 Acknowledgements
- [NLTK Documentation](https://www.nltk.org/)
- [VADER Sentiment Analysis Paper](https://github.com/cjhutto/vaderSentiment)
- Python & Open Source Community
