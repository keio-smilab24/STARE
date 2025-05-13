import os
import uuid

import numpy as np
import torch
from pydantic import BaseModel
from PIL import Image

from .clip_embedder import clip_embeddder
from .GoogleVisionOCR import GoogleVisionOCR
from .Kanavip import Kanavip
from .model import ClipReverie
from .MultilayerDINOv2 import MultilayerDINOv2
from .openai_api import AskToOpenaiChatCompletion, InstElements
from .Patchfy import PatchClipEmbeddder, PatchDinov2Embeddder
from .stella_embedder import stella_embedder
from .TextTokenizer import NETokenizer, OCRTokenizer


class InstructionFeats(BaseModel):
    instruction_id: str
    instruction_clip: np.ndarray
    instruction_stella: np.ndarray
    target_object_noun_stella: np.ndarray
    target_object_explanation_stella: np.ndarray
    ne_tokens: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class ImageFeats(BaseModel):
    image_id: str
    image_clip: np.ndarray
    gpt4o_stella: np.ndarray
    patch_clip: np.ndarray
    patch_dinov2: np.ndarray
    multilayer_dinov2: np.ndarray
    ocr_tokens: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class InstructionResult(BaseModel):
    instruction_embeddings: np.ndarray
    instruction_id: str
    ne_tokens: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class ImageResult(BaseModel):
    image_embeddings: np.ndarray
    image_id: str
    ocr_tokens: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class FeatureExtractor:

    def __init__(self, model="gpt-4o", max_tokens=1024, ckpt_path=None):
        assert ckpt_path is not None, "ckpt_path is required"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model = ClipReverie(
            patch_x=3,
            patch_y=3,
            default_w=0.6,
        ).cuda(self.device)
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

        self.clip_embedder = clip_embeddder()
        self.stella_embedder = stella_embedder()
        self.ask_to_openai = AskToOpenaiChatCompletion(
            model=model, max_tokens=max_tokens
        )
        self.ne_tokenizer = NETokenizer()
        self.ocr_tokenizer = OCRTokenizer()
        self.google_vision_ocr = GoogleVisionOCR(
            credentials_path=os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )
        self.patch_clip_embedder = PatchClipEmbeddder(patch_x=3, patch_y=3)
        self.patch_dinov2_embedder = PatchDinov2Embeddder(patch_x=3, patch_y=3)
        self.kanavip = Kanavip(
            model=model,
            max_tokens=max_tokens,
            x_num=0,
            y_num=0,
            mark_overwrap_threshold=0,
            mark_font_size=20,
            mark_transparency=128,
            mark_lang="kana",
            mark_num=10,
            font_path="data/ipag.ttf",
        )
        self.multilayer_dinov2 = MultilayerDINOv2(
            use_cls=False, use_layers=[-1, -7, -13, -19]
        )

    def embed_instruction(self, inst) -> InstructionResult:
        instruction_feats: InstructionFeats = self.process_instruction(inst)
        with torch.no_grad():
            instruction_embeddings = self.model.text_encoder(
                torch.tensor(instruction_feats.instruction_clip)
                .to(self.device)
                .unsqueeze(0),
                torch.tensor(instruction_feats.instruction_stella)
                .to(self.device)
                .unsqueeze(0),
                torch.tensor(instruction_feats.target_object_explanation_stella)
                .to(self.device)
                .unsqueeze(0),
            )
        return InstructionResult(
            instruction_embeddings=instruction_embeddings.detach().cpu().numpy(),
            instruction_id=instruction_feats.instruction_id,
            ne_tokens=np.expand_dims(instruction_feats.ne_tokens, axis=0),
        )

    def embed_image(self, image: Image) -> ImageResult:
        iamge_feats: ImageFeats = self.process_image(image)
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(
                torch.tensor(iamge_feats.image_clip).to(self.device).unsqueeze(0),
                torch.tensor(iamge_feats.gpt4o_stella).to(self.device).unsqueeze(0),
                torch.tensor(iamge_feats.patch_clip).to(self.device).unsqueeze(0),
                torch.tensor(iamge_feats.patch_dinov2).to(self.device).unsqueeze(0),
                torch.tensor(iamge_feats.multilayer_dinov2)
                .to(self.device)
                .unsqueeze(0),
            )
        return ImageResult(
            image_embeddings=image_embeddings.detach().cpu().numpy(),
            image_id=iamge_feats.image_id,
            ocr_tokens=np.expand_dims(iamge_feats.ocr_tokens, axis=0),
        )

    def process_instruction(self, inst) -> InstructionFeats:
        instruction_id = str(uuid.uuid4())

        instruction_clip = self.clip_embedder.embed_text(inst)
        instruction_stella = self.stella_embedder.embed_text(inst)

        inst_elements: InstElements = self.ask_to_openai.process_target_noun(inst)
        target_object_noun = inst_elements.english_target_object_noun
        target_object_explanation = inst_elements.english_target_object_explanation
        named_entities = inst_elements.english_named_entities

        target_object_noun_stella = self.stella_embedder.embed_text(target_object_noun)
        target_object_explanation_stella = self.stella_embedder.embed_text(
            target_object_explanation
        )
        ne_tokens = self.ne_tokenizer.encode_text(named_entities)

        return InstructionFeats(
            instruction_id=instruction_id,
            instruction_clip=instruction_clip,
            instruction_stella=instruction_stella,
            target_object_noun_stella=target_object_noun_stella,
            target_object_explanation_stella=target_object_explanation_stella,
            ne_tokens=ne_tokens,
        )

    def process_image(self, image: Image) -> ImageFeats:
        image_id = str(uuid.uuid4())

        image_clip: np.ndarray = self.clip_embedder.embed_image(image)
        patch_clip: np.ndarray = self.patch_clip_embedder.embed_image(image)
        patch_dinov2: np.ndarray = self.patch_dinov2_embedder.embed_image(image)

        ocr_results = self.google_vision_ocr.perform_ocr(image)
        ocr_texts = [ocr_result["description"] for ocr_result in ocr_results]
        ocr_tokens = self.ocr_tokenizer.encode_text(ocr_texts)

        kanavip_result = self.kanavip.embed_image(image, ocr_results)
        gpt4o_stella = self.stella_embedder.embed_text(kanavip_result)

        multilayer_dinov2 = self.multilayer_dinov2.embed_image(image)

        return ImageFeats(
            image_id=image_id,
            image_clip=image_clip,
            gpt4o_stella=gpt4o_stella,
            patch_clip=patch_clip,
            patch_dinov2=patch_dinov2,
            multilayer_dinov2=multilayer_dinov2,
            ocr_tokens=ocr_tokens,
        )

    def calc_scores(
        self, image_embeddings, text_embeddings, image_tokens, text_tokens
    ) -> list[float]:
        with torch.no_grad():
            logits = self.model.calc_logits(
                torch.tensor(image_embeddings).to(self.device).float(),
                torch.tensor(text_embeddings).to(self.device).float(),
                torch.tensor(image_tokens).to(self.device).long(),
                torch.tensor(text_tokens).to(self.device).long(),
            )
        return logits.detach().cpu().tolist()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    model = FeatureExtractor()
    inst = "Pass me the box written butt paste next to the blue desitin box on the top shelf."
    image_path = "data/0.jpg"
    image_feats: ImageFeats = model.embed_image(image_path)

    exit()
