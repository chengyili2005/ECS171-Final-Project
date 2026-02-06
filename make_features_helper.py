from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

"""
Helper functions to create certain features
"""

nltk.download('punkt')
nltk.download('vader_lexicon')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
DEFAULT_MODEL = DistilBertModel.from_pretrained('distilbert-base-uncased')
DEFAULT_MODEL.to(DEVICE)
SIA = SentimentIntensityAnalyzer()


# Inputs: Only training dataset

def tfidfvec(train_data: list[str], ngram_range=(1,2)):
  """
  Takes in a train dataset and returns the TfidfVectorizer fitted on that data
  """
  vectorizer = TfidfVectorizer(ngram_range = ngram_range , max_df= 0.95, min_df = 5, max_features = 1000)
  transformed = vectorizer.fit_transform(train_data)
  return transformed, vectorizer

def countvec(train_data: list[str], ngram_range=(1,2)):
  """
  Takes in a train dataset and returns the CountVectorizer fitted on that data
  """
  vectorizer = CountVectorizer(ngram_range = ngram_range, max_features = 1000)
  transformed = vectorizer.fit_transform(train_data)
  return transformed, vectorizer

# Inputs: A single piece of text / single sample

def get_embeddings(text, embedding_model=DEFAULT_MODEL, tokenizer=DEFAULT_TOKENIZER):
  """
  Takes in a single piece of text and returns the last hidden state from a pretrained model
  """
  tokenized_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
  tokenized_text = {k: v.to(DEVICE) for k, v in tokenized_text.items()}
  with torch.no_grad():
    outputs = embedding_model(**tokenized_text)
  embeddings = outputs.last_hidden_state[:, 0, :][0].cpu()
  del tokenized_text
  del outputs
  torch.cuda.empty_cache()

  return embeddings

def word_ratio(text: str, provided_words):
  """
  Takes in a single piece of text and counts for the ratio of words in that text that are provided_words
  """
  words = text.split()
  count = 0
  for word in words:
    if word in provided_words:
      count += 1
  return count/max(len(words), 1)

def reps_ratio(text: str):
  """
  Takes in a single piece of text and counts how many words are repeated
  """
  words = text.split()
  unique_words = set()
  for word in words:
    unique_words.add(word)
  return 1 - len(unique_words) / max(len(words), 1)

def character_count(text_sample):
  """
  Takes in a single piece of text and counts how many characters
  """
  return len(text_sample)

def word_count(text_sample):
  """
  Takes in a single piece of text and counts how many words
  """
  return len(text_sample.split())

def avg_word_length(text_sample):
  """
  Takes in a single piece of text and returns the average word length
  """
  words = [word for word in text_sample.split()]
  return sum([len(word) for word in words])/max(len(words), 1)

def sia_sentiment(text_sample):
  """
  Takes in a single piece of text and returns its sia sentiment polarity socre
  """
  return SIA.polarity_scores(text_sample)['compound']



# Inputs: a list of sentences from a single sample

def avg_sentence_length_in_words(sentences: list[str]):
  """
  Takes in a list of sentences and returns an average sentence length in words
  """
  return sum([len(sentence.split()) for sentence in sentences])/max(len(sentences), 1)

def avg_sentence_length_in_characters(sentences: list[str]):
  """
  Takes in a list of sentences and returns an average sentence length in characters
  """
  return sum([len(sentence) for sentence in sentences])/max(len(sentences), 1)

