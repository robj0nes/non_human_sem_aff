import base64
import json
from pathlib import Path
from io import BytesIO
import PIL.Image
import httpx
import anthropic

# local imports
from models.foundation_model import AffordanceMap, FoundationModel

root = Path(__file__).parent.parent


class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # Remove the 'proxies' argument if present
        super().__init__(*args, **kwargs)


class Claude(FoundationModel):
    def __init__(self, model_variant: str, key_file_path: str, system_prompt: str = None):
        super().__init__(model_class="claude", system_prompt=system_prompt)
        self.endpoint = "anthropic"
        self.model_variant = model_variant
        keyfile = json.load(open(key_file_path))
        if 'anthropic' not in keyfile.keys():
            raise ValueError(
                "Could not find Anthropic key in keyfile. Please ensure a key is provided under 'anthropic'")
        self.key = keyfile['anthropic']
        self.client = anthropic.Anthropic(
            api_key=keyfile['anthropic'],
            http_client=CustomHTTPClient()
        )

    def get_model_response(self, message: str, image_path: str = None, system_prompt: str = None) -> AffordanceMap | str | None:
        tool = {
            "name": 'AffordanceMap',
            "description": 'A map of objects to robot affordances, according to the system message',
            "input_schema": AffordanceMap.model_json_schema()
        }

        if image_path is not None:
            image = PIL.Image.open(image_path)
            im_file = BytesIO()
            image.save(im_file, format='PNG')
            im_bytes = im_file.getvalue()
            im_b64 = base64.b64encode(im_bytes).decode('utf-8')
            content = [
                        {
                            'type': 'image',
                            'source': {
                                'type': "base64",
                                'media_type': 'image/png',
                                'data': im_b64
                            }
                        },
                        {
                            'type': 'text',
                            'text': message,
                        }
                    ]
        else:
            content = [{
                            'type': 'text',
                            'text': message,
                        }]

        message = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            system=system_prompt if system_prompt is not None else self.system_prompt,
            tools=[tool],
            messages=[
                {
                    'role': 'user',
                    'content': content
                }
            ]
        )

        structured_response = None
        for content in message.content:
            if content.type == 'tool_use':
                structured_response = content.input
        if structured_response is not None:
            try:
                valid = AffordanceMap.model_validate(structured_response)
                return valid
            except:
                return f"Failed to convert: {structured_response}"
        else:
            return f"No content in response"
