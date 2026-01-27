import base64
import json
from openai import OpenAI

from models.foundation_model import AffordanceMap, FoundationModel, CustomHTTPClient


class GPT(FoundationModel):
    def __init__(self, model_variant: str, key_file_path: str, system_prompt: str = None):
        super().__init__(model_class="gpt", system_prompt=system_prompt)
        self.endpoint = "openai"
        self.model_variant = model_variant
        keyfile = json.load(open(key_file_path))
        if 'openai' not in keyfile.keys():
            raise ValueError("Could not find OpenAI key in keyfile. Please ensure a key is provided under 'openai'")
        self.openai_key = keyfile['openai']
        self.client = OpenAI(api_key=self.openai_key, http_client=CustomHTTPClient())

    def get_model_response(self, message: str, image_path: str, system_prompt: str) -> AffordanceMap:
        base64_image = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
        response = self.client.responses.parse(
            model=self.model_variant,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{system_prompt}"
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{message}"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }],
            text_format=AffordanceMap
        )
        return response.output_parsed
