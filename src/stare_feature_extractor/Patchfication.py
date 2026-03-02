import clip
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class Patchfy:
    def __init__(self, patch_x=3, patch_y=3):
        self.patch_x = patch_x
        self.patch_y = patch_y

    def get_patch_images(self, image: Image) -> list:
        img_w, img_h = image.size
        cell_w = img_w // (self.patch_x + 1)
        cell_h = img_h // (self.patch_y + 1)
        patch_images = []
        for idy in range(self.patch_y):
            for idx in range(self.patch_x):
                left = cell_w * idx
                upper = cell_h * idy
                right = min(img_w, left + cell_w * 2)
                lower = min(img_h, upper + cell_h * 2)
                patch_images.append(image.crop((left, upper, right, lower)))
        return patch_images


class PatchClipEmbeddder(Patchfy):
    def __init__(self, patch_x=3, patch_y=3):
        super().__init__(patch_x, patch_y)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        clip_base_model = "ViT-L/14"
        self.clip_model, self.preprocess_clip = clip.load(
            clip_base_model, device=self.device
        )

    def embed_image(self, image: str):
        patch_images = self.get_patch_images(image)
        image_features = np.empty((0, 768))
        for patch_image in patch_images:
            with torch.no_grad():
                image_feature = self.clip_model.encode_image(
                    self.preprocess_clip(patch_image).unsqueeze(0).to(self.device)
                )
            image_feature = image_feature.detach().cpu().numpy()
            image_features = np.vstack((image_features, image_feature))

        return image_features


class PatchDinov2Embeddder(Patchfy):

    def __init__(self, patch_x=3, patch_y=3, use_cls=True):
        super().__init__(patch_x, patch_y)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(self.device)
        self.model.eval()
        self.model.config.output_hidden_states = True
        self.use_cls = use_cls

    def embed_image(self, image: str):
        patch_images = self.get_patch_images(image)
        image_features = np.empty((0, 1024))

        for patch_image in patch_images:
            inputs = self.processor(images=patch_image, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            if self.use_cls:
                features = last_hidden_states[:, 0, :]
            else:
                features = last_hidden_states[:, 1:, :].mean(dim=1)
            features = features.cpu().numpy()
            image_features = np.vstack((image_features, features))

        return image_features
