import json

import PIL.Image
from google import genai
from models.foundation_model import AffordanceMap, FoundationModel


# TODO: Gemini can provide bounding box detection - test this.

class Gemini(FoundationModel):
    def __init__(self, model_variant: str, key_file_path: str, system_prompt: str = None, get_bounding_boxes=False):
        super().__init__(model_class="gemini", system_prompt=system_prompt)
        self.endpoint = "open_api"
        self.model_variant = model_variant
        self.system_prompt = system_prompt
        keyfile = json.load(open(key_file_path))
        if 'gemini' not in keyfile.keys():
            raise ValueError("Could not find Gemini key in keyfile. Please ensure a key is provided under 'gemini'")
        self.key = keyfile['gemini']
        self.client = genai.Client(api_key=self.key)
        self.get_bbs = get_bounding_boxes

    def get_model_response(self, message: str, image_path: str = None, system_prompt: str = None) -> AffordanceMap:
        if image_path is not None:
            img = PIL.Image.open(image_path)
            bb_req = " Provide bounding boxes for each detection." if self.get_bbs else ""
            content = [
                img,
                system_prompt + message + bb_req if system_prompt is not None else self.system_prompt + message + bb_req
            ]
        else:
            content = [
                system_prompt + message if system_prompt is not None else self.system_prompt + message
            ]
        response = self.client.models.generate_content(
            model=self.model_variant,
            contents=content,
            config={
                'response_mime_type': 'application/json',
                'response_schema': AffordanceMap,
            },
        )
        return response.parsed
