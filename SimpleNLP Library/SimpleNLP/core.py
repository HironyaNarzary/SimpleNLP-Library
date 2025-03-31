import re
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

# Download required NLTK data with error handling
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("punkt_tab")  # Additional resource needed for tokenization


class SimpleNLP:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove user @ references and '#' from text
        text = re.sub(r"\@\w+|\#", "", text)
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def tokenize_words(self, text):
        """Word tokenization"""
        return word_tokenize(text)

    def tokenize_sentences(self, text):
        """Sentence tokenization"""
        return sent_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove stopwords from tokenized text"""
        return [word for word in tokens if word not in self.stop_words]

    def stem_words(self, tokens):
        """Stem words using Porter Stemmer"""
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize_words(self, tokens):
        """Lemmatize words using WordNet"""
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def get_word_frequencies(self, tokens):
        """Calculate word frequencies"""
        return Counter(tokens)

    def plot_word_frequency(self, tokens, top_n=20):
        """Plot word frequency distribution"""
        freq_dist = FreqDist(tokens)
        freq_dist.plot(top_n, title="Word Frequency Distribution")
        plt.show()

    def get_most_common_words(self, tokens, n=10):
        """Get most common words"""
        freq_dist = FreqDist(tokens)
        return freq_dist.most_common(n)

    def get_word_stats(self, tokens):
        """Get basic statistics about the text"""
        vocab_size = len(set(tokens))
        total_words = len(tokens)
        avg_word_length = sum(len(word) for word in tokens) / total_words
        lexical_diversity = vocab_size / total_words

        return {
            "vocabulary_size": vocab_size,
            "total_words": total_words,
            "average_word_length": avg_word_length,
            "lexical_diversity": lexical_diversity,
        }

    def preprocess_text(self, text, steps=["clean", "tokenize", "remove_stopwords"]):
        """Pipeline for text preprocessing"""
        processed = text

        if "clean" in steps:
            processed = self.clean_text(processed)

        if "tokenize" in steps:
            processed = self.tokenize_words(processed)

        if "remove_stopwords" in steps:
            processed = self.remove_stopwords(processed)

        if "stem" in steps:
            processed = self.stem_words(processed)

        if "lemmatize" in steps:
            processed = self.lemmatize_words(processed)

        return processed


# Example usage with error handling
if __name__ == "__main__":
    try:
        nlp = SimpleNLP()

        sample_text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and human language.
        It focuses on how to program computers to process and analyze large amounts of natural language data.
        """

        # Clean and tokenize
        cleaned_text = nlp.clean_text(sample_text)
        tokens = nlp.tokenize_words(cleaned_text)
        print("Tokens:", tokens)

        # Remove stopwords
        filtered_tokens = nlp.remove_stopwords(tokens)
        print("Filtered Tokens:", filtered_tokens)

        # Lemmatize
        lemmatized = nlp.lemmatize_words(filtered_tokens)
        print("Lemmatized:", lemmatized)

        # Frequency analysis
        freq = nlp.get_word_frequencies(lemmatized)
        print("Frequencies:", freq)

        # Plot word frequency (only if matplotlib is available)
        try:
            nlp.plot_word_frequency(lemmatized)
        except Exception as e:
            print(f"Could not plot frequencies: {e}")

        # Get statistics
        stats = nlp.get_word_stats(lemmatized)
        print("Statistics:", stats)

        # Using the preprocessing pipeline
        processed = nlp.preprocess_text(
            sample_text, steps=["clean", "tokenize", "remove_stopwords", "lemmatize"]
        )
        print("Pipeline Output:", processed)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure all required NLTK data is downloaded.")
