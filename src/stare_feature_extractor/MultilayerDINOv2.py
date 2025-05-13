import json
import os

import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class MultilayerDINOv2:
    def __init__(self, use_cls: bool = True, use_layers: list = [-1]):
        model_name = "facebook/dinov2-large"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model.config.output_hidden_states = True
        self.use_cls = use_cls
        self.use_layers = use_layers

    def embed_image(self, image: Image):
        result = np.empty((0, 1024))
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        all_hidden_states = outputs.hidden_states

        for layer_index in self.use_layers:
            last_hidden_states = all_hidden_states[layer_index]
            if self.use_cls:
                features = last_hidden_states[:, 0, :]
            else:
                features = last_hidden_states[:, 1:, :].mean(dim=1)
            features = features.squeeze(0)
            features = features.cpu().numpy()
            result = np.vstack((result, features))

        return result


if __name__ == "__main__":
    embedder = MultilayerDINOv2(use_cls=False, use_layers=[-1, -7, -13, -19])

    database_path_list = [
        "data/textcaps/textcaps_database.json",
    ]
    output_dir = "multilayer_dinov2"

    for database_path in database_path_list:
        database_name = database_path.split("/")[1]

        print(f"\nProcessing {database_name}")
        database = json.load(open(database_path))
        for item in tqdm.tqdm(database, desc="Processing Items"):
            image_path_list = item["image_path"]
            full_image_feature_path_list = item["full_image_feature_path"]
            if len(image_path_list) != len(full_image_feature_path_list):
                raise ValueError(
                    "image_path_list and full_image_feature_path_list should have the same length."
                )
            for image_path, full_image_feature_path in zip(
                image_path_list, full_image_feature_path_list
            ):
                output_path = full_image_feature_path.replace(
                    "image_features", output_dir
                )
                dinov2_feature = embedder.embed_image(image_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, dinov2_feature)
