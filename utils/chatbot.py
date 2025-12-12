# utils/chatbot.py
import re
from typing import Optional, Dict, Any

try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False


def _fuzzy_best_match(query: str, choices: list, threshold: int = 70) -> Optional[str]:
    """Return best matching choice for query using RapidFuzz (preferred) or difflib."""
    q = query.lower()
    choices_lower = [c.lower() for c in choices]

    if _HAS_RAPIDFUZZ:
        best = process.extractOne(q, choices_lower, scorer=fuzz.partial_ratio)
        if best and best[1] >= threshold:
            idx = choices_lower.index(best[0])
            return choices[idx]
        return None
    else:
        # difflib fallback
        matches = difflib.get_close_matches(q, choices_lower, n=1, cutoff=threshold/100.0)
        if matches:
            idx = choices_lower.index(matches[0])
            return choices[idx]
        return None


def find_title_in_query(query: str, df, id_col: str) -> Optional[str]:
    """
    Versi lanjutan — lebih pintar menemukan judul dari query:
    Steps (in order):
      1. Exact full match (case-insensitive)
      2. Substring match (title contained in query)
      3. Fuzzy matching (rapidfuzz / difflib)
      4. Word-overlap heuristic
    Returns the original title string from dataframe if found, otherwise None.
    """
    if not query or query.strip() == "":
        return None

    q = query.lower()
    titles = df[id_col].astype(str).tolist()
    titles_lower = [t.lower() for t in titles]

    # 1. Exact full match
    for original, tl in zip(titles, titles_lower):
        if tl == q:
            return original

    # 2. Substring (judul ada di query)
    for original, tl in zip(titles, titles_lower):
        if tl in q:
            return original

    # 3. Fuzzy match
    fuzzy = _fuzzy_best_match(q, titles, threshold=68)
    if fuzzy:
        return fuzzy

    # 4. Word overlap (heuristic)
    words = re.findall(r"[\w'-]+", q)
    if not words:
        return None

    best_candidate = None
    best_score = 0
    for original, tl in zip(titles, titles_lower):
        t_words = re.findall(r"[\w'-]+", tl)
        if not t_words:
            continue
        common = set(words) & set(t_words)
        score = len(common) / float(len(t_words))
        if score > best_score and score >= 0.4:
            best_score = score
            best_candidate = original
    if best_candidate:
        return best_candidate

    return None


def generate_chatbot_response(query: str, engine, df, id_col: str, topn: int = 5, weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Chatbot lanjutan:
      - Mencoba mendeteksi judul dari query
      - Mengembalikan rekomendasi jika ditemukan
      - Memberikan saran / fallback bila tidak ditemukan

    Returns dict dengan keys:
      - title_detected: Optional[str]
      - recommendations: pd.DataFrame (jika ada)
      - message: fallback message (jika tidak ada)
    """
    title = find_title_in_query(query, df, id_col)
    if title:
        try:
            recs = engine.recommend_by_title(title, topn=topn, weights=weights)
            return {
                "title_detected": title,
                "recommendations": recs
            }
        except Exception as e:
            return {
                "title_detected": title,
                "message": f"Terjadi kesalahan saat membuat rekomendasi: {e}"
            }
    else:
        # helpful suggestions: show top popular / random items and tips
        tips = (
            "Aku tidak menemukan judul dalam dataset. Coba sebutkan judul persisnya, "
            "atau pilih dari daftar item yang ada."
        )
        # example suggestions: first 5 rows
        try:
            examples = df[id_col].astype(str).head(5).tolist()
        except Exception:
            examples = []
        return {
            "title_detected": None,
            "message": tips,
            "examples": examples
        }


class ConversationManager:
    """
    Simple in-memory conversation manager to keep short session context.
    Stores last N queries and last detected title.
    """
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []  # list of (query, detected_title)

    def add(self, query: str, detected_title: Optional[str]):
        self.history.append((query, detected_title))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def last_detected_title(self) -> Optional[str]:
        for q, t in reversed(self.history):
            if t:
                return t
        return None

    def get_history(self):
        return list(self.history)
