"""Text preprocessing utilities for NLP tasks."""

import re
import string
import pandas as pd
from typing import List, Dict, Optional
import logging

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Comprehensive text preprocessing for job descriptions and skill extraction."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        self.stop_words = STOP_WORDS
        self.tech_stopwords = self._get_tech_stopwords()

    def _get_tech_stopwords(self) -> set:
        """Get technology-specific stop words that should be removed."""
        return {
            "experience", "years", "year", "work", "working", "job", "position",
            "role", "team", "company", "business", "industry", "looking", "seeking",
            "required", "preferred", "must", "should", "able", "strong", "good",
            "excellent", "knowledge", "skills", "ability", "proficiency"
        }

    def clean_text(self, text: str, preserve_technical_terms: bool = True) -> str:
        """Clean and normalize text while preserving important technical information."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive punctuation but preserve some for technical terms
        if preserve_technical_terms:
            # Keep dots in version numbers (e.g., Python 3.8, Node.js)
            text = re.sub(r'\.(?!\d)', ' ', text)
            # Keep + in technical terms (e.g., C++, Angular2+)
            text = re.sub(r'\+(?![+\w])', ' ', text)
            # Keep # in technical terms (e.g., C#, F#)
            text = re.sub(r'#(?!\w)', ' ', text)
        else:
            # Remove all punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Final cleanup
        text = text.strip()

        return text

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        if not self.nlp:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [sent.strip() for sent in sentences if sent.strip()]

        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def extract_technical_phrases(self, text: str) -> List[str]:
        """Extract technical phrases and terms."""
        # Patterns for common technical phrases
        technical_patterns = [
            r'\b\w+\.\w+\b',  # Versioned technologies (e.g., Python 3.8)
            r'\b\w+\+\+?\b',  # Plus-based technologies (e.g., C++)
            r'\b\w+#\b',      # Hash-based technologies (e.g., C#)
            r'\b\w+\.js\b',   # JavaScript frameworks
            r'\b\w+SQL\b',    # SQL variants
            r'\bAPI\w*\b',    # API-related terms
            r'\b\w*-\w+\b',   # Hyphenated terms
        ]

        technical_terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technical_terms.extend(matches)

        return list(set(technical_terms))

    def tokenize_with_context(self, text: str) -> Dict[str, List[str]]:
        """Tokenize text while maintaining context for technical terms."""
        if not self.nlp:
            # Simple fallback tokenization
            tokens = text.split()
            return {
                "tokens": tokens,
                "entities": [],
                "noun_phrases": []
            }

        doc = self.nlp(text)

        return {
            "tokens": [token.text for token in doc if not token.is_space],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
            "lemmas": [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        }

    def preprocess_job_description(self, description: str) -> Dict[str, any]:
        """Comprehensive preprocessing of job descriptions."""
        if pd.isna(description) or not description.strip():
            return {
                "original_length": 0,
                "cleaned_text": "",
                "sentences": [],
                "technical_phrases": [],
                "tokens": [],
                "entities": [],
                "noun_phrases": []
            }

        # Basic cleaning
        cleaned_text = self.clean_text(description, preserve_technical_terms=True)

        # Extract components
        sentences = self.extract_sentences(cleaned_text)
        technical_phrases = self.extract_technical_phrases(cleaned_text)
        tokenization = self.tokenize_with_context(cleaned_text)

        return {
            "original_length": len(description),
            "cleaned_text": cleaned_text,
            "sentences": sentences,
            "technical_phrases": technical_phrases,
            "tokens": tokenization["tokens"],
            "entities": tokenization["entities"],
            "noun_phrases": tokenization["noun_phrases"]
        }

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "description"
    ) -> pd.DataFrame:
        """Preprocess text data in a DataFrame."""
        logger.info(f"Preprocessing text data in column: {text_column}")

        if text_column not in df.columns:
            logger.error(f"Column {text_column} not found in DataFrame")
            return df

        processed_data = []

        for idx, row in df.iterrows():
            text = row[text_column]
            preprocessing_result = self.preprocess_job_description(text)

            # Add preprocessing results as new columns
            processed_row = row.copy()
            processed_row[f"{text_column}_cleaned"] = preprocessing_result["cleaned_text"]
            processed_row[f"{text_column}_length"] = preprocessing_result["original_length"]
            processed_row[f"{text_column}_sentences_count"] = len(preprocessing_result["sentences"])
            processed_row[f"{text_column}_technical_phrases"] = ", ".join(preprocessing_result["technical_phrases"])

            processed_data.append(processed_row)

            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} descriptions")

        result_df = pd.DataFrame(processed_data)
        logger.info(f"Completed preprocessing for {len(result_df)} records")

        return result_df
