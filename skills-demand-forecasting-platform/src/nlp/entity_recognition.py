"""Entity Recognition Module"""
import spacy

class EntityRecognizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        # TODO: Implement entity recognition
        return []
