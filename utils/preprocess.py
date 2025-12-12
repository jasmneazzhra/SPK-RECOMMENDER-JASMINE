import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler




def prepare_features(df, text_cols=None, num_cols=None):
	"""
	Membangun feature matrix:
	- gabungkan kolom teks menjadi satu kolom 'combined_text'
	- buat tfidf matrix jika ada text_cols
	- scale numeric features jika ada num_cols
	Mengembalikan dict berisi tfidf_vectorizer, tfidf_matrix, scaler, num_matrix
	"""
	res = {}
	n = len(df)

	if text_cols:
		df['combined_text'] = df[text_cols].astype(str).agg(' '.join, axis=1)
		tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
		tfidf_matrix = tfidf.fit_transform(df['combined_text'])
		res['tfidf'] = tfidf
		res['tfidf_matrix'] = tfidf_matrix
	else:
		res['tfidf'] = None
		res['tfidf_matrix'] = None

	if num_cols:
		scaler = StandardScaler()
		num_matrix = scaler.fit_transform(df[num_cols].astype(float).fillna(0.0))
		res['scaler'] = scaler
		res['num_matrix'] = num_matrix
	else:
		res['scaler'] = None
		res['num_matrix'] = None

	return res