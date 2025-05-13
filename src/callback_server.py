import base64
import io
import json
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing_extensions import Annotated


def start(conf, callback, cb_embed_image, cb_embed_inst, cb_inference):
    fapi = FastAPI()

    @fapi.post(
        "/oneshot/all",
        responses={200: {"content": {"application/json": {"example": {}}}}},
    )
    def execute_oneshot(
        image: UploadFile = File(..., description="Image to be evaluated"),
        instruction: UploadFile = File(
            ..., description="JSON file containing instructions"
        ),
    ):
        input_json = json.load(instruction.file)
        ret = callback(image.file, input_json["instruction"], input_json["mode"])
        return JSONResponse(content={"score": ret})

    @fapi.post(
        "/oneshot/with_embedded_imgs",
        responses={200: {"content": {"application/json": {"example": {}}}}},
    )
    def execute_oneshot_with_embedded_imgs(
        image: Annotated[
            List[UploadFile], File(description="Embedded images to be evaluated")
        ],
        instruction: UploadFile = File(
            ..., description="JSON file containing instructions"
        ),
    ):
        input_json = json.load(instruction.file)
        embedded_inst = cb_embed_inst(
            input_json["instruction"], input_json["mode"], np_return=True
        )
        embedded_img = np.concatenate([np.load(img.file) for img in image])
        ret = cb_inference(embedded_img, embedded_inst, num_k=20)
        return JSONResponse(content={"scores": ret})

    @fapi.post(
        "/embedding/image",
        responses={200: {"content": {"application/json": {"example": {}}}}},
    )
    def execute_embedding_image(
        image: UploadFile = File(..., description="Image to be embedded"),
    ):
        embedded_img = cb_embed_image(image.file, np_return=True)
        buf = io.BytesIO()
        np.save(buf, embedded_img)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return JSONResponse(content={"embeddings": b64})

    host_name = conf["hostname"]
    port_num = int(conf["port"])

    uvicorn.run(fapi, host=host_name, port=port_num)
