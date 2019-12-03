# 
import re
import pickle
# 
from sklearn.feature_extraction.text import HashingVectorizer

stop_words_pkl_path = "Model/stopwords.pkl"

def tokenizer(text):
	stop_words = pickle.load(open(stop_words_pkl_path, 'rb'))
	#
	text = re.sub('<[^>]*>', '', text)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
	text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
	tokenized = [words for words in text.split() if words not in stop_words]
	return tokenized

vectorizer = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)