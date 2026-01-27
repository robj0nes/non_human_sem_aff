import warnings
from abc import ABC, abstractmethod
from typing import List, Dict

import httpx
import numpy as np
import torch
from pydantic import BaseModel, Field


from robot.robot_localisation import RobotLocalisation


class Affordance(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str = Field(description="Name of the affordance", alias='name')
    objects: List[str] | None = Field(description="A list of objects with this affordance",
                                      alias="objects")


class ObjectWorldData(BaseModel):
    name: str = Field(description="Name of the object", alias="name")
    location: List[float] = Field(description="3D coordinates of an object in world space", alias="location")
    affordances: List[Affordance] = Field(description="List of affordances identified for this object",
                                          alias="affordances")


class LabelledFrame(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    frame_id: int = Field(description="The frame number recorded from the robot", alias="frame_id")
    annotated_image: np.ndarray = Field(description="The image data used for bounding_box annotation",
                                        alias="annotated_image")
    bounding_boxes: torch.Tensor = Field(description="The bounding boxes returned from GroundingDINO",
                                         alias="bounding_boxes")
    object_list: List[str] = Field(description="The object list provided to GroundingDINO", alias="object_list")


class ObjectLabelledFrame(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    frame_id: int = Field(description="The frame number recorded from the robot", alias="frame_id")
    annotated_image: np.ndarray = Field(description="The image data used for bounding_box annotation",
                                        alias="annotated_image")
    bounding_box: torch.Tensor = Field(description="The bounding box for this specific object", alias="bounding_box")
    odometry: RobotLocalisation | None = Field(description="The localisation of the robot at the nearest time-stamped "
                                                    "record to the frame timestamp", alias="odometry")


class ObjectMap(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str = Field(description="Name of the object", alias="name")
    frames: List[ObjectLabelledFrame | None] = Field(description="A list of frames and bounding boxes for this object",
                                                     alias="frames")
    estimate: np.ndarray | None = Field(description="Estimated object position in global reference frame",
                                        alias="estimate")


class AffordanceMap(BaseModel):
    objects: List[str] | None = Field(description="List of all objects detected.", alias="objects")
    affordances: List[Affordance] | None = Field(description="List of all affordances", alias="affordances")


class AffordanceResponse(BaseModel):
    endpoint: str = Field(description="Name of the model endpoint being used", alias="endpoint")
    model: str = Field(description="Name of the model variant being used", alias="model")
    affordance: AffordanceMap | None = Field(description="The map of affordance-object relations", alias="affordance")
    object_localisation: List[ObjectMap] | None = Field(description="The object localisation specifics",
                                                        alias="object_localisation")


class VisionDetection(BaseModel):
    endpoint: str = Field(description="Name of the endpoint used for the vision based object detection",
                          alias="endpoint")
    model: str = Field(description="Name of the model being used", alias="model")
    objects: List[str] | None = Field(description='Objects detected by the vision model', alias="objects")
    lm_response: List[AffordanceResponse] | None = Field(description='Affordance response from the language model',
                                                         alias='lm_response')


class PromptResponse(BaseModel):
    prompt_id: str = Field(description="ID of the prompt, enumeration of prompt used (0 if none), "
                                       "should have the format {VLM_sys}_{VLM_usr}_{LM_sys}_{LM_usr}",
                           alias="prompt_id")
    user_prompt: List[str | None] = Field(description="The prompt(s) provided to the foundation the user level. "
                                                      "Ordered by stack in pipeline (eg. vlm->lm)",
                                          alias="user_prompt")
    system_prompt: List[str | None] = Field(description="The prompt(s) provided to the foundation model at the "
                                                        "system level. Ordered by stack in pipeline (eg. vlm->lm)",
                                            alias="system_prompt")
    affordances: List[AffordanceResponse | VisionDetection] | None = Field(
        description="List of affordance responses according to the pipeline hierarchy",
        alias="affordances")


class Frame(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    pipeline: str = Field(description="Name of the pipeline used for the experiment", alias="pipeline")
    frame_num: int = Field(description="Framer number being tested", alias="frame_num")
    frame: np.ndarray = Field(description="The image data used for the experiment", alias="frame")
    prompt_responses: List[PromptResponse] | None = Field(
        description="List of affordance responses, given a prompt variant")

class FrameIteration(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    frames: List[Frame] | None = Field(description="List of all frames in the iteration", alias="frames")
    frame_num: int | None = Field(description="The frame number being tested", alias="frame_num")




class Experiment(BaseModel):
    """
    Main experiment results class. Contains a list of Frames - each with respective endpoint test results.
    And a set of world maps seperated by endpoint test.
    """

    class Config:
        arbitrary_types_allowed = True

    frames: List[FrameIteration] | None = Field(description="List of all frames in the experiment", alias="frames")
    world_map: Dict[str, Dict[str, List[ObjectWorldData]] | None] | None = Field(
        description="The world map of the objects", alias="world_map")


class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # Remove the 'proxies' argument if present
        super().__init__(*args, **kwargs)


class FoundationModel(ABC):
    def __init__(self, model_class=None, system_prompt=None):
        if system_prompt is None:
            warnings.warn("No system prompt set. It is advisable to prep the LLM for the task context.")
        self.system_prompt = system_prompt
        self.model_class = model_class

    @abstractmethod
    def get_model_response(self, message: str, image_path: str, system_prompt: str) -> AffordanceMap:
        raise NotImplementedError

    def update_system_prompt(self, prompt):
        self.system_prompt = prompt
