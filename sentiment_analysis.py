# Load dataset
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
import contractions
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import pandas as pd
df = pd.read_csv('./IMDB-Dataset.csv')

# Remove duplicate rows
df = df.drop_duplicates()

nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))

# Expanding contractions


def expand_contractions(text):
    return contractions.fix(text)

# Function to clean data


def preprocess_text(text):
    wl = WordNetLemmatizer()
    soup = BeautifulSoup(text, "html.parser")   # Removing html tags
    text = soup.get_text()
    # Expanding chatwords and contracts clearing contractions
    text = expand_contractions(text)
    emoji_clean = re.compile("["
                             u"\ U0001F600 -\ U0001F64F "
                             u"\ U0001F300 -\ U0001F5FF "
                             u"\ U0001F680 -\ U0001F6FF "
                             u"\ U0001F1E0 -\ U0001F1FF "
                             u"\ U00002702 -\ U000027B0 "
                             u"\ U000024C2 -\ U0001F251 "
                             "]+", flags=re. UNICODE)
    text = emoji_clean.sub(r'', text)
    text = re.sub(r'\.(?=\S)', '. ', text)  # add space after full stop
    text = re.sub(r'http\S+', '', text)  # remove usls
    text = "".join([
        word.lower() for word in text if word not in string.punctuation
    ])  # remove punctuation and make text lowercase
    text = " ".join([
        wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()])
    return text


df['review'] = df['review'].apply(preprocess_text)

# creating autopt arguments


def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)


freg_pos = len(df[df['sentiment'] == 'positive'])
freg_neg = len(df[df['sentiment'] == 'negative'])

data = [freg_pos, freg_neg]

labels = ['positive', 'negative']
# Create pie chart
pie, ax = plt.subplots(figsize=[11, 7])
plt.pie(x=data, autopct=lambda pct: func(pct, data), explode=[0.0025]*2,
        pctdistance=0.5, colors=[sns.color_palette()[0], 'tab:red'], textprops={'fontsize': 16})

# plt.title( ’ Frequencies of sentiment labels ’, fontsize =14 , fontweight = ’ bold ’)
labels = [r'Positive', r'Negative']
plt.legend(labels, loc="best", prop={'size': 14})
pie.savefig("PieChart.png")
plt.show()

print(f'Number of positive reviews: {freg_pos}')
print(f'Number of negative reviews: {freg_neg}')

words_len = df['review'].str.split().map(lambda x: len(x))
df_temp = df.copy()
df_temp['words length'] = words_len

hist_positive = sns.displot(
    data=df_temp[df_temp['sentiment'] == 'positive'],
    x="words length", hue="sentiment", kde=True, height=7, aspect=1.1, legend=False
).set(title='Words in positive reviews')

hist_negative = sns.displot(
    data=df_temp[df_temp['sentiment'] == 'negative'],
    x="words length", hue="sentiment", kde=True, height=7, aspect=1.1, legend=False, palette=['red']
).set(title='Words in negative reviews')

plt.figure(figsize=(7, 7.1))
kernel_distibution_number_words_plot = sns.kdeplot(
    data=df_temp, x="words length", hue="sentiment", fill=True, palette=[sns.color_palette()[0], 'red']
).set(title='Words in reviews')
plt.legend(title='Sentiment', labels=['negative', 'positive'])
plt.show()

# Split train and test dataset
label_encode = LabelEncoder()
x_data = label_encode.fit_transform(df['review'])
y_data = label_encode.fit_transform(df['sentiment'])

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42)

# Vectorize data
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
tfidf_vectorizer.fit(df['review'])

x_train_encoded = tfidf_vectorizer.transform(df['review'].iloc[x_train])
y_train_encoded = tfidf_vectorizer.transform(df['review'].iloc[y_train])
x_test_encoded = tfidf_vectorizer.transform(df['review'].iloc[x_test])
y_test_encoded = tfidf_vectorizer.transform(df['review'].iloc[y_test])

# Train and evaluate model
dt_classifier = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)
dt_classifier.fit(x_train_encoded, y_train)
y_perd = dt_classifier.predict(x_test_encoded)
accuracy_of_dt = accuracy_score(y_perd, y_test)
print(f'Accuracy: {accuracy_of_dt}')

rf_classifier = RandomForestClassifier(
    random_state=42
)

rf_classifier.fit(x_train_encoded, y_train)
y_pred = rf_classifier.predict(x_test_encoded)
accuracy_of_rf = accuracy_score(y_perd, y_test)
print(f'Accuracy: {accuracy_of_rf}')