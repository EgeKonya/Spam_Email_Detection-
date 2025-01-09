import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Download stopwords if not already available
nltk.download('stopwords')

# Function to clean and preprocess the text data
def preprocess_text(text, stemmer, stopwords_set):
    # Remove newline characters
    text = text.replace('\r\n', ' ').lower()
    # Remove punctuation and split into words
    words = text.translate(str.maketrans('', '', string.punctuation)).split()
    # Stem words and remove stopwords
    processed_words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(processed_words)

# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Initialize stemmer and stopwords set
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Preprocess the 'text' column
df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, stemmer, stopwords_set))

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Target variable
y = df['label_num']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model performance
print("Random Forest model's accuracy:", clf.score(X_test, y_test))
