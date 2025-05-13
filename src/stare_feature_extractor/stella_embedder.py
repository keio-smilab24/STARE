import warnings

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class stella_embedder:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with warnings.catch_warnings():
            self.model = SentenceTransformer(
                "dunzhang/stella_en_400M_v5", trust_remote_code=True
            ).to(self.device)

    def embed_text(self, text) -> np.array:
        result = self.model.encode(text)
        return np.array(result)
