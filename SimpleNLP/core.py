import re
import string
from collections import Counter
from typing import List, Dict, Union, Set, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist


def _download_nltk_data():
    """Download required NLTK data with robust error handling."""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt_tab", "punkt_tab")
    ]
    
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name)
            except Exception as e:
                raise RuntimeError(f"Failed to download NLTK resource {name}: {str(e)}")


# Ensure NLTK data is available at module level
_download_nltk_data()


class SimpleNLP:
    """A simple NLP processor for basic text analysis tasks.
    
    Features:
    - Text cleaning (URLs, mentions, punctuation, etc.)
    - Word and sentence tokenization
    - Stopword removal
    - Stemming and lemmatization
    - Word frequency analysis
    - Basic text statistics
    
    Args:
        language: Language for stopwords (default: 'english')
        remove_numbers: Whether to remove numbers during cleaning (default: True)
        remove_punct: Whether to remove punctuation during cleaning (default: True)
    """
    
    def __init__(self, language: str = 'english', remove_numbers: bool = True, remove_punct: bool = True):
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            raise ValueError(f"Unsupported language: {language}. Available languages: {stopwords.fileids()}")
            
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.remove_numbers = remove_numbers
        self.remove_punct = remove_punct
        self.language = language

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted elements.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text with specified elements removed
        """
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove user @ references and '#' from text
        text = re.sub(r"\@\w+|\#", "", text)
        
        if self.remove_punct:
            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Convert to lowercase
        text = text.lower()
        
        if self.remove_numbers:
            # Remove numbers
            text = re.sub(r"\d+", "", text)
            
        # Remove extra whitespace and trim
        text = " ".join(text.split()).strip()
        return text

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of word tokens
        """
        return word_tokenize(text)

    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence tokens
        """
        return sent_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokenized text.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [word for word in tokens if word not in self.stop_words]

    def stem_words(self, tokens: List[str]) -> List[str]:
        """Stem words using Porter Stemmer.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize_words(self, tokens: List[str]) -> List[str]:
        """Lemmatize words using WordNet.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    @staticmethod
    def get_word_frequencies(tokens: List[str]) -> Counter:
        """Calculate word frequencies.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Counter object with word frequencies
        """
        return Counter(tokens)

    def plot_word_frequency(self, tokens: List[str], top_n: int = 20) -> None:
        """Plot word frequency distribution.
        
        Args:
            tokens: List of word tokens
            top_n: Number of top words to display
            
        Raises:
            ImportError: If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )
        
        freq_dist = FreqDist(tokens)
        freq_dist.plot(top_n, title="Word Frequency Distribution")
        plt.show()

    @staticmethod
    def get_most_common_words(tokens: List[str], n: int = 10) -> List[tuple]:
        """Get most common words.
        
        Args:
            tokens: List of word tokens
            n: Number of top words to return
            
        Returns:
            List of tuples (word, count) for most common words
        """
        freq_dist = FreqDist(tokens)
        return freq_dist.most_common(n)

    @staticmethod
    def get_word_stats(tokens: List[str]) -> Dict[str, Union[int, float]]:
        """Get basic statistics about the text.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Dictionary containing:
            - vocabulary_size: Number of unique words
            - total_words: Total word count
            - average_word_length: Average word length
            - lexical_diversity: Type-token ratio
        """
        vocab_size = len(set(tokens))
        total_words = len(tokens)
        avg_word_length = sum(len(word) for word in tokens) / total_words if total_words > 0 else 0
        lexical_diversity = vocab_size / total_words if total_words > 0 else 0

        return {
            "vocabulary_size": vocab_size,
            "total_words": total_words,
            "average_word_length": avg_word_length,
            "lexical_diversity": lexical_diversity,
        }

    def preprocess_text(
        self,
        text: str,
        steps: Optional[List[str]] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """Pipeline for text preprocessing.
        
        Args:
            text: Input text to process
            steps: List of processing steps to apply. Default steps:
                   ["clean", "tokenize", "remove_stopwords"]
            **kwargs: Additional arguments for processing steps
            
        Returns:
            Processed text (str if no tokenization, list of tokens otherwise)
            
        Raises:
            ValueError: If invalid steps are provided
        """
        if steps is None:
            steps = ["clean", "tokenize", "remove_stopwords"]
            
        valid_steps = {"clean", "tokenize", "remove_stopwords", "stem", "lemmatize"}
        invalid_steps = set(steps) - valid_steps
        if invalid_steps:
            raise ValueError(f"Invalid steps: {invalid_steps}. Valid options: {valid_steps}")
            
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


if __name__ == "__main__":
    # Example usage
    try:
        nlp = SimpleNLP(language='english')

        sample_text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and human language.
        It focuses on how to program computers to process and analyze large amounts of natural language data.
        """

        # Demonstrate cleaning
        cleaned = nlp.clean_text(sample_text)
        print(f"Cleaned text:\n{cleaned}\n")

        # Full preprocessing pipeline
        processed = nlp.preprocess_text(
            sample_text,
            steps=["clean", "tokenize", "remove_stopwords", "lemmatize"]
        )
        print(f"Processed tokens:\n{processed}\n")

        # Frequency analysis
        freq = nlp.get_word_frequencies(processed)
        print(f"Top 5 words:\n{freq.most_common(5)}\n")

        # Statistics
        stats = nlp.get_word_stats(processed)
        print("Text statistics:")
        for k, v in stats.items():
            print(f"{k:>20}: {v:.2f}" if isinstance(v, float) else f"{k:>20}: {v}")

        # Plotting (commented out by default)
        # nlp.plot_word_frequency(processed)

    except Exception as e:
        print(f"Error: {e}")
