import io
import json
import os

import tqdm
from google.cloud import vision
from PIL import Image


class GoogleVisionOCR:
    def __init__(self, credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = vision.ImageAnnotatorClient()

    def perform_ocr(self, image: Image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or "PNG")
        content = img_byte_arr.getvalue()

        image = vision.Image(content=content)

        response = self.client.text_detection(image=image)
        texts = response.text_annotations

        texts = texts[1:]

        ocr_results = []
        for text in texts:
            vertices = text.bounding_poly.vertices
            x_coords = [vertex.x for vertex in vertices]
            y_coords = [vertex.y for vertex in vertices]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            ocr_results.append(
                {
                    "description": text.description,
                    "bounds": [
                        {"x": x_min, "y": y_min},
                        {"x": x_max, "y": y_max},
                    ],
                }
            )

        if response.error.message:
            raise Exception(f"{response.error.message}")

        return ocr_results

    @staticmethod
    def demo_print(img_path, ocr_results):
        from PIL import Image, ImageDraw

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for ocr_result in ocr_results:
            description = ocr_result["description"]
            bounds = ocr_result["bounds"]
            x1, y1 = bounds[0]["x"], bounds[0]["y"]
            x2, y2 = bounds[1]["x"], bounds[1]["y"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), description, fill="red")

        img.save("demo_bbox_img.jpg")


if __name__ == "__main__":
    database_path_list = [
        "data/gogetit_reftext/gogetit_reftext_database.json",
        "data/gogetit_instruction/gogetit_instruction_database.json",
    ]
    ocr = GoogleVisionOCR()

    for database_path in database_path_list:
        database = json.load(open(database_path))
        savee_path = database_path.replace(".json", "_ocr.json")
        print(f"Processing {database_path}")
        ocr_results = {}
        if os.path.exists(savee_path):
            ocr_results = json.load(open(savee_path))
        for idx, item in enumerate(tqdm.tqdm(database)):
            image_path_list = item["image_path"]
            for image_path in image_path_list:
                if image_path in ocr_results:
                    continue
                ocr_result = ocr.perform_ocr(image_path)
                ocr_results[image_path] = ocr_result

            if idx % 100 == 0:
                json.dump(ocr_results, open(savee_path, "w"), indent=1)

        json.dump(ocr_results, open(savee_path, "w"), indent=1)
