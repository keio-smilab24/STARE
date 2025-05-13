
import json
import os

import openai
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


class InstElements(BaseModel):
    english_target_object_noun: str
    english_target_object_explanation: str
    english_named_entities: list[str]


class AskToOpenaiBase:
    def __init__(self, model="gpt-4o", max_tokens=1024):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.config = {
            "model": model,
            "max_tokens": max_tokens,
            "n": 1,
            "temperature": 0,
        }

    def ask_once(self, messages, response_format=None):
        config = self.config

        try:
            if response_format:
                config["response_format"] = response_format
                res = self.client.beta.chat.completions.parse(
                    messages=messages, **config
                )
                response = res.choices[0].message.content
                response = json.loads(response)
            else:
                res = self.client.chat.completions.create(messages=messages, **config)
                response = res.choices[0].message.content
            return response
        except openai.RateLimitError:
            print("Request to OpenAI API was timed out.")
            return None

    def create_batch(self, jsonl_path):
        print(f"Uploading {jsonl_path}")
        batch_input_file = self.client.files.create(
            file=open(jsonl_path, "rb"), purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "GPT-4o VIP",
            },
        )
        print(f"Batch ID: {batch.id}")

        return batch.id

    def check_batch_status(self, batch_id):
        batch = self.client.batches.retrieve(batch_id)
        return batch.status

    def wait_batch_completion(self, batch_id_list):
        while True:
            for i, batch_id in enumerate(batch_id_list):
                print(f"[{i}]: Status: {self.check_batch_status(batch_id)}", end="\r")
            if all(
                [
                    self.check_batch_status(batch_id) == "completed"
                    for batch_id in batch_id_list
                ]
            ):
                break
        print("All batch is completed.")

    def download_batch_output(self, batch_id_list):
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        llm_result = {}
        for batch_id in tqdm(batch_id_list, desc="Downloading batch output"):
            batch = self.client.batches.retrieve(batch_id)
            output_file = self.client.files.retrieve(batch.output_file_id)
            file_response = self.client.files.content(output_file.id)

            raw_output_path = os.path.join(tmp_dir, f"{batch_id}.jsonl")
            with open(raw_output_path, "wb") as f:
                f.write(file_response.content)

            with open(raw_output_path, "r") as f:
                output_jsonl = f.readlines()

            output_jsonl = [json.loads(item) for item in output_jsonl]
            batch_output_jsonl = {
                item["custom_id"]: item["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                for item in output_jsonl
            }
            llm_result.update(batch_output_jsonl)

        os.system(f"rm -rf {tmp_dir}")

        return llm_result


class AskToOpenaiChatCompletion(AskToOpenaiBase):
    def __init__(self, model="gpt-4o", max_tokens=1024):
        super().__init__(model=model, max_tokens=max_tokens)

    def process_target_noun(self, instruction) -> InstElements:
        prompt = (
            "Extract named entities from the following instruction. "
            "Named entities include specific names of brands, locations, or unique identifiers, even if they are common words in other contexts. "
            "Additionally, create noun phrases for the target object included in the instruction, as well as descriptions of the object focusing only on its visual characteristics. "
            "Use the knowledge available to the LLM as the information source. "
            "The output should be translated into English."
            f"Instruction: {instruction}"
        )

        messages = [{"role": "user", "content": prompt}]
        llm_output = self.ask_once(messages, response_format=InstElements)
        inst_elements = InstElements(**llm_output)

        return inst_elements
