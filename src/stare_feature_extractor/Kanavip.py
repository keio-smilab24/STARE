import argparse
import base64
import io
import json
import os
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from tqdm import tqdm

from .openai_api import AskToOpenaiBase


class KanavipResult(BaseModel):
    english_image_description: str


class Kanavip(AskToOpenaiBase):

    def __init__(
        self,
        model="gpt-4o",
        max_tokens=1024,
        x_num=4,
        y_num=4,
        cell_per_patch=2,
        mark_overwrap_threshold=0.2,
        mark_font_size: int = 20,
        mark_transparency: int = 128,
        mark_lang: Literal[
            "kana",
            "en_lower",
            "en_upper",
            "en",
            "num",
            "greek",
            "greek_upper",
            "greek_lower",
        ] = "kana",
        mark_num=10,
        font_path="data/ipag.ttf",
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
        )
        self.x_num = x_num
        self.y_num = y_num
        self.cell_per_patch = cell_per_patch
        self.mark_overwrap_threshold = mark_overwrap_threshold
        self.mark_font_size = mark_font_size
        self.mark_transparency = mark_transparency
        self.font_path = font_path
        self.full_font = False
        if mark_lang == "kana":
            self.full_font = True
            self.mark_text_list = [
                "ア",
                "イ",
                "ウ",
                "エ",
                "オ",
                "カ",
                "キ",
                "ク",
                "ケ",
                "コ",
                "サ",
                "シ",
                "ス",
                "セ",
                "ソ",
                "タ",
                "チ",
                "ツ",
                "テ",
                "ト",
                "ナ",
                "ニ",
                "ヌ",
                "ネ",
                "ノ",
                "ハ",
                "ヒ",
                "フ",
                "ヘ",
                "ホ",
                "マ",
                "ミ",
                "ム",
                "メ",
                "モ",
                "ヤ",
                "ユ",
                "ヨ",
                "ラ",
                "リ",
                "ル",
                "レ",
                "ロ",
                "ワ",
                "ヲ",
                "ン",
            ]
        elif mark_lang == "en_lower":
            self.mark_text_list = [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
            ]
        elif mark_lang == "en_upper" or mark_lang == "en":
            self.mark_text_list = [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
            ]
        elif mark_lang == "num":
            self.mark_text_list = [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]
        elif mark_lang == "greek" or mark_lang == "greek_upper":
            self.full_font = True
            self.mark_text_list = [
                "Α",
                "Β",
                "Γ",
                "Δ",
                "Ε",
                "Ζ",
                "Η",
                "Θ",
                "Ι",
                "Κ",
                "Λ",
                "Μ",
                "Ν",
                "Ξ",
                "Ο",
                "Π",
                "Ρ",
                "Σ",
                "Τ",
                "Υ",
                "Φ",
                "Χ",
                "Ψ",
                "Ω",
            ]
        elif mark_lang == "greek_lower":
            self.full_font = True
            self.mark_text_list = [
                "α",
                "β",
                "γ",
                "δ",
                "ε",
                "ζ",
                "η",
                "θ",
                "ι",
                "κ",
                "λ",
                "μ",
                "ν",
                "ξ",
                "ο",
                "π",
                "ρ",
                "σ",
                "τ",
                "υ",
                "φ",
                "χ",
                "ψ",
                "ω",
            ]
        else:
            raise ValueError("mark_lang must be jp, en_lower, en_upper, or en")
        self.mark_lang = mark_lang
        self.mark_num = mark_num

    def create_llm_messages(self, image: Image.Image, detection_results_texts: list):
        indexed_ocr_text_list = []
        for idx, detection_result_text in enumerate(detection_results_texts):
            description = detection_result_text["description"]
            indexed_ocr_text_list.append(f"{self.mark_text_list[idx]}: {description}")

        prompt = (
            "Describe the objects in this image in detail, including their positional relationships to surrounding objects. "
            f"The positional labels in the image and the OCR-detected text pairs are as follows: {', '.join(indexed_ocr_text_list)}. "
            "The position label with white text on a black background is the center of the OCR text detection area and the top outside the area. "
            "If multiple texts are associated with a single positional label, identify the correct text for that label. "
            "Do not explain using position labels, but use them only to understand the image. "
            "Output the description in English."
        )

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ]

        return messages

    def get_llm_desc(self, image: Image.Image, detection_results_texts: list) -> str:
        messages = self.create_llm_messages(image, detection_results_texts)
        llm_desc: KanavipResult = self.ask_once(messages, response_format=KanavipResult)
        return llm_desc.english_image_description

    def create_patch_items(
        self, image, detection_results_texts: list, ocr_bbox=False
    ):
        patch_items = self._get_patch_image(image)
        patch_items = self._add_ocr_to_patch(patch_items, detection_results_texts)
        patch_items = self._integrate_ocr(patch_items)
        patch_items = self._filter_detected_texts(patch_items)
        patch_items = self._set_of_mark(patch_items, ocr_bbox)
        return patch_items

    def _get_patch_image(
        self,
        image: str,
    ):
        img_size = image.size

        patch_items = []
        original_item = {
            "x_start": 0,
            "y_start": 0,
            "x_end": img_size[0],
            "y_end": img_size[1],
            "image": image,
        }
        patch_items.append(original_item)

        if self.x_num == 0 or self.y_num == 0:
            return patch_items

        cell_size = (img_size[0] // self.x_num, img_size[1] // self.y_num)
        patch_size = (
            int(cell_size[0] * self.cell_per_patch),
            int(cell_size[1] * self.cell_per_patch),
        )
        patch_x_num = self.x_num - self.cell_per_patch + 1
        patch_y_num = self.y_num - self.cell_per_patch + 1

        for patch_y in range(patch_y_num):
            for patch_x in range(patch_x_num):
                patch_item = {
                    "x_start": patch_x * cell_size[0],
                    "y_start": patch_y * cell_size[1],
                    "x_end": patch_x * cell_size[0] + patch_size[0],
                    "y_end": patch_y * cell_size[1] + patch_size[1],
                }
                if patch_x == patch_x_num - 1:
                    patch_item["x_end"] = img_size[0]
                if patch_y == patch_y_num - 1:
                    patch_item["y_end"] = img_size[1]
                patch_image = image.crop(
                    (
                        patch_item["x_start"],
                        patch_item["y_start"],
                        patch_item["x_end"],
                        patch_item["y_end"],
                    )
                )
                patch_item["image"] = patch_image

                patch_items.append(patch_item)

        return patch_items

    def _add_ocr_to_patch(self, patch_items, detection_results_texts: list):
        for idx, patch_item in enumerate(patch_items):
            image = patch_item["image"]
            x_start = patch_item["x_start"]
            y_start = patch_item["y_start"]
            x_end = patch_item["x_end"]
            y_end = patch_item["y_end"]
            patch_items[idx]["detection_results_texts"] = []

            for detection_result_text in detection_results_texts:
                description = detection_result_text["description"]
                bounds = detection_result_text["bounds"]
                bound_x_start = bounds[0]["x"]
                bound_y_start = bounds[0]["y"]
                bound_x_end = bounds[1]["x"]
                bound_y_end = bounds[1]["y"]

                if (
                    x_start <= bound_x_start
                    and bound_x_end <= x_end
                    and y_start <= bound_y_start
                    and bound_y_end <= y_end
                ):
                    relative_bound_x_start = bound_x_start - x_start
                    relative_bound_y_start = bound_y_start - y_start
                    relative_bound_x_end = bound_x_end - x_start
                    relative_bound_y_end = bound_y_end - y_start
                    relative_bounds = [
                        {
                            "x": relative_bound_x_start,
                            "y": relative_bound_y_start,
                        },
                        {
                            "x": relative_bound_x_end,
                            "y": relative_bound_y_end,
                        },
                    ]
                    patch_items[idx]["detection_results_texts"].append(
                        {
                            "description": [description],
                            "bounds": relative_bounds,
                        }
                    )

        return patch_items

    def get_mark_bbox(self, ocr_bbox, image_size):
        if self.full_font:
            mark_x_len = self.mark_font_size
        else:
            mark_x_len = self.mark_font_size // 2
        mark_y_len = self.mark_font_size
        ocr_x_center = (ocr_bbox[0]["x"] + ocr_bbox[1]["x"]) / 2
        mark_x_start = int(ocr_x_center - mark_x_len / 2)
        mark_y_start = max(int(ocr_bbox[0]["y"] - mark_y_len), 0)

        mark_bbox = [
            {
                "x": mark_x_start,
                "y": mark_y_start,
            },
            {
                "x": mark_x_start + mark_x_len,
                "y": mark_y_start + mark_y_len,
            },
        ]
        return mark_bbox

    def check_overwrap(self, bbox1, bbox2, mark_bbox1, mark_bbox2) -> bool:
        bbox1_area = (bbox1[1]["x"] - bbox1[0]["x"]) * (bbox1[1]["y"] - bbox1[0]["y"])
        mark_bbox1_area = (mark_bbox1[1]["x"] - mark_bbox1[0]["x"]) * (
            mark_bbox1[1]["y"] - mark_bbox1[0]["y"]
        )

        x1 = max(bbox1[0]["x"], bbox2[0]["x"])
        y1 = max(bbox1[0]["y"], bbox2[0]["y"])
        x2 = min(bbox1[1]["x"], bbox2[1]["x"])
        y2 = min(bbox1[1]["y"], bbox2[1]["y"])
        if x1 >= x2 or y1 >= y2:
            pass
        else:
            overwrap_area = (x2 - x1) * (y2 - y1)
            if overwrap_area / bbox1_area >= self.mark_overwrap_threshold:
                return True

        x1 = max(bbox1[0]["x"], mark_bbox2[0]["x"])
        y1 = max(bbox1[0]["y"], mark_bbox2[0]["y"])
        x2 = min(bbox1[1]["x"], mark_bbox2[1]["x"])
        y2 = min(bbox1[1]["y"], mark_bbox2[1]["y"])
        if x1 >= x2 or y1 >= y2:
            pass
        else:
            overwrap_area = (x2 - x1) * (y2 - y1)
            if overwrap_area / bbox1_area >= self.mark_overwrap_threshold:
                return True

        x1 = max(mark_bbox1[0]["x"], bbox2[0]["x"])
        y1 = max(mark_bbox1[0]["y"], bbox2[0]["y"])
        x2 = min(mark_bbox1[1]["x"], bbox2[1]["x"])
        y2 = min(mark_bbox1[1]["y"], bbox2[1]["y"])
        if x1 >= x2 or y1 >= y2:
            pass
        else:
            overwrap_area = (x2 - x1) * (y2 - y1)
            if overwrap_area / mark_bbox1_area >= self.mark_overwrap_threshold:
                return True

        x1 = max(mark_bbox1[0]["x"], mark_bbox2[0]["x"])
        y1 = max(mark_bbox1[0]["y"], mark_bbox2[0]["y"])
        x2 = min(mark_bbox1[1]["x"], mark_bbox2[1]["x"])
        y2 = min(mark_bbox1[1]["y"], mark_bbox2[1]["y"])
        if x1 >= x2 or y1 >= y2:
            pass
        else:
            overwrap_area = (x2 - x1) * (y2 - y1)
            if overwrap_area / mark_bbox1_area >= self.mark_overwrap_threshold:
                return True

        return False

    def integrate_bbox(self, bbox1, bbox2):
        x1 = min(bbox1[0]["x"], bbox2[0]["x"])
        y1 = min(bbox1[0]["y"], bbox2[0]["y"])
        x2 = max(bbox1[1]["x"], bbox2[1]["x"])
        y2 = max(bbox1[1]["y"], bbox2[1]["y"])
        return [
            {
                "x": x1,
                "y": y1,
            },
            {
                "x": x2,
                "y": y2,
            },
        ]

    def _integrate_ocr(self, patch_items) -> list:
        for idx, patch_item in enumerate(patch_items):
            detection_results_texts = patch_item["detection_results_texts"]

            enable_index_list = [True] * len(detection_results_texts)
            for i, detection_result_text1 in enumerate(detection_results_texts):
                description1 = detection_result_text1["description"]
                if not enable_index_list[i]:
                    continue
                ocr_bbox1 = detection_result_text1["bounds"]
                mark_bbox1 = self.get_mark_bbox(ocr_bbox1, patch_item["image"].size)
                for j, detection_result_text2 in enumerate(detection_results_texts):
                    description2 = detection_result_text2["description"]
                    if i == j:
                        continue
                    if not enable_index_list[j]:
                        continue
                    ocr_bbox2 = detection_result_text2["bounds"]
                    mark_bbox2 = self.get_mark_bbox(ocr_bbox2, patch_item["image"].size)
                    is_overwrap = self.check_overwrap(
                        ocr_bbox1, ocr_bbox2, mark_bbox1, mark_bbox2
                    )
                    if is_overwrap:
                        ocr_bbox1 = self.integrate_bbox(ocr_bbox1, ocr_bbox2)
                        detection_results_texts[i]["bounds"] = ocr_bbox1
                        detection_results_texts[i][
                            "description"
                        ] += detection_results_texts[j]["description"]
                        enable_index_list[j] = False

            detection_results_texts = [
                detection_results_texts[i]
                for i in range(len(detection_results_texts))
                if enable_index_list[i]
            ]

            patch_items[idx]["detection_results_texts"] = detection_results_texts

        return patch_items

    def _filter_detected_texts(self, patch_items):
        for idx, patch_item in enumerate(patch_items):
            detection_results_texts = patch_item["detection_results_texts"]
            detection_results_texts = sorted(
                detection_results_texts,
                key=lambda x: (x["bounds"][1]["x"] - x["bounds"][0]["x"])
                * (x["bounds"][1]["y"] - x["bounds"][0]["y"]),
                reverse=True,
            )
            patch_items[idx]["detection_results_texts"] = detection_results_texts[
                : self.mark_num
            ]

            patch_items[idx]["detection_results_texts"] = sorted(
                patch_items[idx]["detection_results_texts"],
                key=lambda x: x["bounds"][0]["x"],
            )
        return patch_items

    def _set_of_mark(self, patch_items, ocr_bbox=False) -> list:
        for idx, patch_item in enumerate(patch_items):
            image = patch_item["image"].convert("RGBA")
            overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            for idy, detection_result_text in enumerate(
                patch_item["detection_results_texts"]
            ):
                bounds = detection_result_text["bounds"]
                mark_bbox = self.get_mark_bbox(bounds, image.size)
                mark_text = self.mark_text_list[idy]
                overlay_draw.rectangle(
                    (
                        mark_bbox[0]["x"],
                        mark_bbox[0]["y"],
                        mark_bbox[1]["x"],
                        mark_bbox[1]["y"],
                    ),
                    fill=(0, 0, 0, self.mark_transparency),
                )
                overlay_draw.text(
                    (mark_bbox[0]["x"], mark_bbox[0]["y"]),
                    mark_text,
                    fill=(255, 255, 255),
                    font=ImageFont.truetype(self.font_path, self.mark_font_size),
                )
                if ocr_bbox:
                    overlay_draw.rectangle(
                        (
                            bounds[0]["x"],
                            bounds[0]["y"],
                            bounds[1]["x"],
                            bounds[1]["y"],
                        ),
                        outline=(173, 255, 47),
                        width=2,
                    )
            image = Image.alpha_composite(image, overlay)

            patch_items[idx]["image"] = image

        return patch_items

    def embed_image(self, image, detection_results_texts):
        patch_items = self.create_patch_items(
            image, detection_results_texts, ocr_bbox=False
        )
        entire_image_item = patch_items[0]
        image = entire_image_item["image"]

        detection_results_texts = entire_image_item["detection_results_texts"]
        messages = self.create_llm_messages(image, detection_results_texts)
        llm_desc = self.ask_once(messages, response_format=KanavipResult)
        kanavip_desc = KanavipResult(**llm_desc)
        return kanavip_desc.english_image_description
