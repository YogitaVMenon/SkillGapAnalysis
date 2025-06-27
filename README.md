AI-Based Skill Gap Analysis System

This project is a comprehensive AI solution that identifies skill gaps in candidate profiles, matches them to relevant job roles, and recommends personalized Coursera courses for upskilling. It uses NLP, Sentence Transformers and evaluation metrics to build a smart, interactive system that supports both real and synthetic data testing.

Key Features

- Skill Matching: Match candidate skills against job requirements using semantic similarity.
- Skill Gap Detection: Identify missing skills from job descriptions.
- Course Recommendation: Recommend top-matching Coursera courses for the missing skills using BERT embeddings.
- System Evaluation: Generate synthetic candidates, simulate positions, compute F1 scores, confusion matrix, and satisfaction metrics.
- Visualization: Generate performance plots and heatmaps for insights.

Technologies & Libraries

- Python
- pandas, NumPy
- spaCy (NLP)
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- Matplotlib, Seaborn
- CSV I/O
- Evaluation Metrics (Precision, Recall, F1, Relevance)

How It Works

1. Input: Resume and a job description
2. Extraction: NLP extracts key skills from both
3. Embedding: BERT model encodes the text to vectors
4. Comparison: Cosine similarity checks overlap and differences
5. Output: Highlights missing skills and suggests courses



