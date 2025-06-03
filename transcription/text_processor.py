import logging
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self):
        self._download_nltk_data()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Academic stop words to remove
        self.academic_stopwords = {
            "um",
            "uh",
            "ah",
            "er",
            "you know",
            "like",
            "so",
            "okay",
            "alright",
            "well",
            "now",
            "today",
            "here",
        }
        self.stop_words.update(self.academic_stopwords)

    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
            nltk.data.find("corpora/wordnet")
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download("punkt")
            nltk.download("stopwords")
            nltk.download("wordnet")

    def process_text(self, text):
        """
        Process the extracted text for better note generation

        Args:
            text (str): Raw text from transcription

        Returns:
            str: Processed text ready for note generation
        """
        logger.info("Processing text...")

        # Step 1: Clean and normalize text
        cleaned_text = self._clean_text(text)

        # Step 2: Remove filler words and improve sentences
        improved_text = self._remove_fillers(cleaned_text)

        # Step 3: Split into sentences and filter
        sentences = sent_tokenize(improved_text)
        processed_sentences = []

        for sentence in sentences:
            # Skip very short sentences (likely transcription errors)
            if len(sentence.split()) < 3:
                continue

            # Process individual sentence
            processed_sentence = self._process_sentence(sentence)
            if processed_sentence:
                processed_sentences.append(processed_sentence)

        # Step 4: Reconstruct text with better paragraph structure
        final_text = self._create_paragraphs(processed_sentences)

        logger.info("Text processing completed")
        return final_text

    def _clean_text(self, text):
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix common transcription issues
        text = re.sub(r"\b(\w)\1{2,}\b", r"\1\1", text)  # Remove repeated letters
        text = re.sub(r"\b[a-zA-Z]\b", "", text)  # Remove single letters

        # Standardize punctuation
        text = re.sub(r"[,]{2,}", ",", text)
        text = re.sub(r"[.]{2,}", ".", text)

        return text.strip()

    def _remove_fillers(self, text):
        """Remove filler words and expressions common in lectures"""
        filler_patterns = [
            r"\b(um|uh|ah|er)\b",
            r"\byou know\b",
            r"\bI mean\b",
            r"\bbasically\b",
            r"\bactually\b",
            r"\bokay so\b",
            r"\balright so\b",
            r"\bso um\b",
        ]

        for pattern in filler_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _process_sentence(self, sentence):
        """Process individual sentence"""
        # Skip sentences that are too repetitive or unclear
        words = word_tokenize(sentence.lower())

        # Remove sentences with too many stop words
        content_words = [
            w for w in words if w not in self.stop_words and w not in string.punctuation
        ]
        if len(content_words) < 2:
            return None

        # Basic sentence reconstruction
        sentence = sentence.strip()
        if not sentence.endswith((".", "!", "?")):
            sentence += "."

        return sentence

    def _create_paragraphs(self, sentences):
        """Group sentences into logical paragraphs"""
        if not sentences:
            return ""

        paragraphs = []
        current_paragraph = []

        for i, sentence in enumerate(sentences):
            current_paragraph.append(sentence)

            # Create paragraph breaks based on:
            # 1. Every 4-6 sentences
            # 2. Topic transitions (detected by certain keywords)
            if len(current_paragraph) >= 4 and (
                i == len(sentences) - 1
                or self._is_topic_transition(
                    sentence, sentences[i + 1] if i + 1 < len(sentences) else ""
                )
            ):
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []

        # Add remaining sentences
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        return "\n\n".join(paragraphs)

    def _is_topic_transition(self, current_sentence, next_sentence):
        """Detect if there's a topic transition between sentences"""
        transition_words = [
            "now",
            "next",
            "moving on",
            "let's",
            "another",
            "furthermore",
            "however",
            "on the other hand",
            "in contrast",
            "meanwhile",
        ]

        next_lower = next_sentence.lower()
        return any(word in next_lower for word in transition_words)
