import clip
import numpy as np
import torch


class clip_embeddder:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        clip_base_model = "ViT-L/14"
        self.clip_model, self.preprocess_clip = clip.load(
            clip_base_model, device=self.device
        )

    def embed_image(self, image, detach=True) -> np.array:
        with torch.no_grad():
            image = self.preprocess_clip(image).unsqueeze(0).to(self.device)
            image_feature = self.clip_model.encode_image(image)
        if detach:
            image_feature = image_feature.detach().cpu().numpy()
        return image_feature.squeeze()

    def embed_text(self, text, detach=True) -> np.array:
        tokenized_text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized_text)
        if detach:
            text_features = text_features.detach().cpu().numpy()
        return text_features.squeeze()
