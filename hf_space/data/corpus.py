import json
import random
from typing import Any, Dict, List


class DocumentCorpus:
    """Loads and manages the document corpus for training and evaluation."""

    def __init__(self, data_file: str = "data/corpus.json", split: str = "train"):
        self.data_file = data_file
        self.split = split
        self.documents = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                all_docs = json.load(f)
            # Filter by split
            docs = [doc for doc in all_docs if doc.get("split", "train") == self.split]
            if not docs:
                print(f"Warning: No documents found for split '{self.split}'.")
            return docs
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Corpus file {self.data_file} not found. Please run 'python data/generator.py' first."
            )

    def sample(self) -> Dict[str, Any]:
        """Returns a random document from the corpus."""
        if not self.documents:
            raise ValueError(f"Corpus is empty for split {self.split}.")
        return random.choice(self.documents)

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all documents in this split (useful for eval)."""
        return self.documents

