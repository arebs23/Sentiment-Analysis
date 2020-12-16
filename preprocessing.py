import re
import emoji
from sklearn.base import BaseEstimator,TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
import string
import pandas as pd


class TextCounts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))
    
    def fit(self, X, y=None, **fit_params):
        # fit method is used when specific operations need to be done on the train data, but not on the test data
        return self
    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x)) 
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
        count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
        count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))
        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))
        # We will replace the emoji symbols with a description, which makes using a regex for counting easier
        # Moreover, it will result in having more words in the tweet
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
        
        df = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags
                           , 'count_capital_words': count_capital_words
                           , 'count_excl_quest_marks': count_excl_quest_marks
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                          })
        
        return df
    
class CleanText(BaseEstimator, TransformerMixin):
    
    def __init__(self, default_stemmer = PorterStemmer(), default_stopwords = stopwords.words('english')):
        self.default_stemmer = default_stemmer
        self.default_stopwords = default_stopwords
        
    def tokenize_text(self, X):
        return [w for s in sent_tokenize(X) for w in word_tokenize(s)]

    def remove_special_characters(self, X, characters=string.punctuation):
        tokens = self.tokenize_text(X)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub(' ', t) for t in tokens if t.isalpha() and len(t) > 1]))

    def stem_text(self, X):
        tokens = self.tokenize_text(X)
        return ' '.join([self.default_stemmer.stem(t) for t in tokens])

    def remove_stopwords(self, X):
        tokens = [w for w in self.tokenize_text(X) if w not in self.default_stopwords]
        return ' '.join(tokens)
    
    def clean(self, X):
        text = X.strip(' ')
        text = text.lower()
        text = self.stem_text(text)
        text = self.remove_special_characters(text)
        text = self.remove_stopwords(text)
        return text
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return list(map(self.clean, X))    


class ColumnExtractor(BaseEstimator,TransformerMixin):

    def __init__(self, cols):
        self.cols = cols
    
    def transform(self, X, **transform_params):
        return X[self.cols]
    
    def fit(self, X, y = None, **fit_params):
        return self

