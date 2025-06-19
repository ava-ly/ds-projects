## Data Science and Machine Learning Projects

.. |OK_ICON| image:: https://raw.githubusercontent.com/ava-ly/ds-projects/icon/ok-24.png

### |OK_ICON| 1. Crime Trend Analysis in Los Angeles | [Completed](https://colab.research.google.com/drive/1AqGJYrLT7S_xg3T7Rwdc6l6W-KenIGR9?usp=sharing) | [Streamlit App](https://crime-in-la.streamlit.app)
--

- Exploratory Data Analysis revealed that crime in Los Angeles is highly patterned and predictable to a degree, driven by a combination of temporal, spatial, situational, and victim-related factors. These patterns can be leveraged for more targeted and efficient public safety interventions.
- Random Forest model can predict broad crime categories with moderate overall accuracy (~65%-70%), performing very well for high-volume, distinct categories like 'Violent Crimes' but struggling with rarer or more ambiguous ones.
- Tools: Pandas, Numpy, Matplotlib, Seaborn, Sklearn, Random Forest, SHAP.

### |OK_ICON| 2. Netflix Content Analysis & Predictive Model Development | [Completed](https://colab.research.google.com/drive/195e05q0ZOsTWc9Hy8nVWTmOrEP1AR2B4?usp=sharing)
-----------

- Exploratory Data Analysis revealed a dynamic Netflix catalog, predominantly featuring movies over TV shows. The United States is the primary content producer, followed by countries like India and the UK. Genres like Drama, Comedy, and Thriller are prevalent. IMDb scores show a somewhat normal distribution, with clear distinctions in rating patterns between movies and TV shows, and variations across different genres and release eras. Documentaries, interestingly, emerged as a genre with distinct rating characteristics.
- After initial training, XGBoost (with early stopping using a validation set) demonstrated the most promising performance, achieving a Test R² of approximately ~0.55-0.60 and a Test MAE of around ~0.6-0.7. This indicates the model can explain a significant portion of the variance in IMDb scores. However, the initial XGBoost evaluation using the test set for early stopping might present slightly optimistic scores.
- Tools: Pandas, Numpy, Matplotlib, Seaborn, NLTK, WordCloud, Sklearn, Scipy, Regression (Linear, Ridge, Lasso), Tree-based (XGBoost, Gradient Boosting, and Random Forest).

### |OK_ICON| 3. SpaceX Falcon 9 First Stage Landing Prediction | [Completed](https://colab.research.google.com/drive/1rUMM7Aj3BKhup3LOyAcjsa9kkjU4nWLF?usp=sharing)
-----------

- Exploratory Data Analysis revealed that variables like `LaunchSite`, `Orbit`, `PayloadMass`, and `FlightNumber` (as a proxy for experience/booster block improvements) show potential relationships with landing success and are important candidates for feature engineering and model training. The temporal improvement in success rates also suggests that time-related features or more recent data might be particularly relevant.
- Based on the evaluated metrics (Accuracy, Jaccard Index, and F1-Score) on the unseen test data, the K-Nearest Neighbors (KNN) model demonstrated the most promising performance for predicting Falcon 9 first stage landing success. Logistic Regression and Decision Tree also provided respectable results. SVM, in its current configuration, was less effective.
- Tools: Pandas, Numpy, Matplotlib, Seaborn, Sklearn, Logistic Regression, Support Vector Machine (SVM), Decision Tree, K-Nearest Neighbors (KNN).

### 4. Analysis and Clustering of Top Streamed Songs on Spotify | Ongoing
-----------

Project Goals:
- Perform a comprehensive exploratory data analysis (EDA) on the "Most Streamed Spotify Songs 2024" dataset to uncover trends, patterns, and relationships in modern music success.
- Apply unsupervised clustering techniques to identify distinct "success profiles" of hit songs based on their performance across various streaming and social media platforms.

### 5. A Data-Driven Analysis of YouTube Trends and Sentiment in USA | Ongoing
-----------

Project Goals:
- Characterize Trending Videos through Exploratory Data Analysis (EDA).
- Develop a machine learning regression model to predict the view count of a trending video based on its features.
- Apply Natural Language Processing (NLP) techniques to calculate a sentiment score for video comments.
- Use unsupervised clustering (K-Means) to discover distinct characteristics of trending videos (e.g., "Viral Sensations," "Niche Favorites").
- Conduct a time-series analysis to understand how YouTube trends and category popularity have evolved over time.

---
## Staying Updated with Data Science, Machine Learning and AI

- [Towards Data Science](https://towardsdatascience.com): Offers daily articles on data science, machine learning, and AI, suitable for beginners to advanced users.
- [KDnuggets](https://www.kdnuggets.com): Provides news, articles, and trends in data science and AI, updated regularly with high-quality content.
- [HackerNoon](https://hackernoon.com): Features a variety of AI and machine learning articles.
- [ScienceDaily](https://sciencedaily.com): Aggregates news on the latest AI research.
- [arXiv](https://arxiv.org): A repository for the latest research papers in computer science, including AI and machine learning.
- [Papers with Code](https://paperswithcode.com): Combines research papers with code, ideal for staying updated on advancements.
- [Google AI Blog](https://research.google/blog/): Shares updates from Google’s AI research team.
- [The Batch](https://www.deeplearning.ai/the-batch): A weekly newsletter from deeplearning.ai highlighting practical research and industry news in deep learning.
