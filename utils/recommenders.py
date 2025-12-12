import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class Engine:
    def __init__(self, df, id_col, features):
        self.df = df
        self.id_col = id_col
        self.features = features
        n = len(df)

        # Text similarity
        if self.features.get('tfidf_matrix') is not None:
            self.tfidf_matrix = self.features['tfidf_matrix']
            self.sim_text = cosine_similarity(self.tfidf_matrix)
        else:
            self.tfidf_matrix = None
            self.sim_text = np.zeros((n, n))

        # Numeric similarity
        if self.features.get('num_matrix') is not None:
            self.num_matrix = self.features['num_matrix']
            self.sim_num = cosine_similarity(self.num_matrix)
        else:
            self.num_matrix = None
            self.sim_num = np.zeros((n, n))

        # clustering using combined small feature (if available)
        combined = []
        if self.num_matrix is not None:
            combined.append(self.num_matrix)
        if self.tfidf_matrix is not None:
            # convert to dense small-size (may be heavy on large datasets)
            combined.append(self.tfidf_matrix.toarray())

        if combined:
            comb = np.hstack(combined)
            k = min(8, max(2, len(self.df)//50 + 2))
            self.km = KMeans(n_clusters=k, random_state=42)
            self.cluster_labels = self.km.fit_predict(comb)
        else:
            self.km = None
            self.cluster_labels = np.zeros(len(self.df), dtype=int)

    def recommend_by_index(self, idx, topn=5, weights=None):
        # default weights
        w = {'text':0.5,'num':0.3,'cluster':0.2}
        if weights:
            w.update(weights)
        # compute hybrid score
        score = np.zeros(len(self.df))
        score += w['text'] * self.sim_text[idx]
        score += w['num'] * self.sim_num[idx]
        # cluster bonus: +0.2 for same cluster
        same_cluster = (self.cluster_labels == self.cluster_labels[idx]).astype(float)
        score += w['cluster'] * same_cluster

        # mask itself
        score[idx] = -1
        top_idx = np.argsort(score)[::-1][:topn]
        res = self.df.iloc[top_idx].copy()
        res['score'] = score[top_idx]
        return res

    def recommend_by_title(self, title, topn=5, weights=None):
        title_str = str(title).strip().lower()
        matches = self.df[self.df[self.id_col].astype(str).str.strip().str.lower() == title_str]
        if len(matches)==0:
            # try containment search
            matches = self.df[self.df[self.id_col].astype(str).str.lower().str.contains(title_str)]
        if len(matches)==0:
            raise ValueError("Title not found in dataset")
        idx = matches.index[0]
        return self.recommend_by_index(idx, topn=topn, weights=weights)

def build_models_and_recommend(df, id_col, features):
    """Factory: build Engine and return it (object with method recommend_by_title)
    """
    engine = Engine(df, id_col, features)
    return engine