from keybert import KeyBERT
from keybert.backend._utils import select_backend
import numpy as np
import re
from typing import List, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
basic_stops = stopwords.words('english')
basic_stops = [w for w in basic_stops if "n't" not in w]
basic_stops = [w for w in basic_stops if w not in ['no','not','nor','but']]
stopwords = basic_stops[:]
stopwords += [w.capitalize() for w in basic_stops]
stopwords += [w.upper() for w in basic_stops]


class AspectKeyBERT(KeyBERT):
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = select_backend(model)
    def extract_aspect_keywords(
        self,
        doc: str,
        use_aspect_as_doc_embedding: bool = False,
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        aspect_keywords: List[str] = None,
    ) -> List[Tuple[str, float]]:
        """Extract keywords/keyphrases for a single document
        Arguments:
            doc: The document for which to extract keywords/keyphrases
            use_aspect_as_doc_embedding: If True, will use the given aspect keywords for doc representation
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases
            use_mmr: Whether to use Max Sum Similarity
            use_mmr: Whether to use MMR
            diversity: The diversity of results between 0 and 1 if use_mmr is True
            nr_candidates: The number of candidates to consider if use_maxsum is set to True
            vectorizer: Pass in your own CountVectorizer from scikit-learn
            aspect_keywords: Aspect keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords
        Returns:
            keywords: the top n keywords for a document with their respective distances
                      to the input document
        """
        try:
            # Extract Words
            if candidates is None:
                if vectorizer:
                    count = vectorizer.fit([doc])
                else:
                    """
                    there's one severe problem behind the KeyBert.
                    for example, `. nice day` and `nice day` will may be extracted together, 
                    since they are highly semantically similar
                    """
                    count = CountVectorizer(
                        ngram_range=keyphrase_ngram_range, lowercase=False, tokenizer=word_tokenize, token_pattern=None
                    ).fit([doc])
                candidates = count.get_feature_names()
                def there_is_punc(text):
                    return len(re.findall(r'[，。：；‘’“”;,\.\?\!\(\)\[\]\{\}:\'\"\-\@\#\$\%\^\&\*]',text))
                candidates = [c for c in candidates if not there_is_punc(c)]

            # Extract Embeddings
            if use_aspect_as_doc_embedding:
                assert aspect_keywords is not None, 'You must provide aspect_keywords when use_seed_as_doc_embedding !!!'
                doc_embedding = self.model.embed([" ".join(aspect_keywords)])
            else:
                doc_embedding = self.model.embed([doc])
            candidate_embeddings = self.model.embed(candidates)

            # Guided KeyBERT with seed keywords
            if aspect_keywords is not None and use_aspect_as_doc_embedding == False:
                aspect_embeddings = self.model.embed([" ".join(aspect_keywords)])
                doc_embedding = np.average(
                    [doc_embedding, aspect_embeddings], axis=0, weights=[1, 1]
                )

            # Calculate distances and extract keywords
            if use_mmr:
                keywords = mmr(
                    doc_embedding, candidate_embeddings, candidates, top_n, diversity
                )
            elif use_maxsum:
                keywords = max_sum_similarity(
                    doc_embedding,
                    candidate_embeddings,
                    candidates,
                    top_n,
                    nr_candidates,
                )
            else:
                distances = cosine_similarity(doc_embedding, candidate_embeddings)
                keywords = [
                    (candidates[index], round(float(distances[0][index]), 4))
                    for index in distances.argsort()[0][-top_n:]
                ][::-1]

            return keywords
        except ValueError:
            return []
