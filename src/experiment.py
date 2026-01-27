import json
import logging
import os
import re
import time
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from GroundingDINO.groundingdino.util.inference import load_image, predict, annotate
from models.foundation_model import (
    ObjectLabelledFrame, LabelledFrame, AffordanceResponse, ObjectMap,
    VisionDetection, Experiment, Frame, ObjectWorldData,
    PromptResponse, FrameIteration
)
from models.anthropic import Claude
from models.gemini import Gemini
from models.openai import GPT
from utils import load_config, load_dino_model, Log


# Constants
ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(filename=f'{ROOT}/logs/experiment.log', filemode='w', level=logging.DEBUG)
DOTS_INTRINSICS = np.array([
    [165.1, 0, 160.2],
    [0, 165.45, 121.35],
    [0, 0, 1]
])

# Type aliases for complex structures
EndpointMap = Dict[str, List[Any]]  # e.g., {'vlm_models': [...], 'lm_models': [...]}
OdometryLookup = Dict[str, Dict[int, Any]]


def setup_environment(config) -> None:
    """Sets up environment variables for model storage based on the OS."""
    if config['paths']['hf_home']:
        os.environ["HF_HOME"] = config['paths']['hf_home']

def grounding_dino_bbs(
        dino_model: Any,
        llm_response: Any,
        image_path: Path
) -> Tuple[np.ndarray, Any, List[str]]:
    """
    Query Grounding DINO for bounding boxes based on structured LLM output.

    Args:
        dino_model: The loaded Grounding DINO model instance.
        llm_response: Structured object containing a list of detected objects.
        image_path: Path to the image file.

    Returns:
        Tuple containing the annotated image (numpy array), boxes, and object list.
    """
    if not llm_response.objects:
        # Handle case where no objects are provided to avoid empty string prompt issues
        image_source, image = load_image(str(image_path))
        return np.array(image_source), [], []

    prompt = ' . '.join(obj.title() for obj in llm_response.objects) + '.'
    box_threshold = 0.35
    text_threshold = 0.25

    image_source, image = load_image(str(image_path))

    boxes, logits, object_list = predict(
        model=dino_model,
        image=image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="mps"
    )

    annotated_frame = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=object_list
    )
    return annotated_frame, boxes, object_list


def annotate_affordance_on_image(
        img_path: Path,
        save_dir: Path,
        affordance: Any,
        model_name: str,
        show_images: bool = False
) -> None:
    """
    Overlays affordance text on an image and saves it.

    Args:
        img_path: Path to the source image.
        save_dir: Directory to save the annotated image.
        affordance: The affordance data structure.
        model_name: Name of the model used for naming the file.
        show_images: Whether to display the image using OS viewer.
    """
    img = Image.open(img_path)
    text_overlay = ""

    # Iterate through fields in the affordance object (pydantic model assumed)
    for elem in affordance:
        field_name, field_value = elem
        if field_name != 'objects':
            if field_value:
                for aff in field_value:
                    objects_str = ", ".join(str(x) for x in aff.objects) if aff.objects else "None"
                    text_overlay += f" {aff.name.title()}: {objects_str}\n"
            else:
                text_overlay += "None \n"

    draw = ImageDraw.Draw(img)
    # Note: Hardcoded position and font size; consider making configurable
    draw.text((15, 350), text_overlay, fill=(255, 255, 255), font_size=25)

    save_path = save_dir / f"Aff_{model_name}_{img_path.name}"
    img.save(save_path)

    if show_images:
        img.show("Affordances")


def triangulate_position(
        robot_pos_1: np.ndarray,
        robot_rot_1: np.ndarray,
        robot_pos_2: np.ndarray,
        robot_rot_2: np.ndarray,
        bb_1: np.ndarray,
        bb_2: np.ndarray
) -> Optional[np.ndarray]:
    """
    Estimates 3D position of an object using triangulation from two camera views.

    Args:
        robot_pos_1: Position vector of robot at frame 1.
        robot_rot_1: Quaternion rotation of robot at frame 1.
        robot_pos_2: Position vector of robot at frame 2.
        robot_rot_2: Quaternion rotation of robot at frame 2.
        bb_1: Bounding box in frame 1 [x1, y1, x2, y2].
        bb_2: Bounding box in frame 2 [x1, y1, x2, y2].

    Returns:
        Estimated 3D position (x, y, z) or None if robot hasn't moved.
    """
    if np.allclose(robot_pos_1, robot_pos_2):
        return None

    # Calculate centers of bounding boxes
    bb1_cent = (bb_1[0] + bb_1[2]) / 2, (bb_1[1] + bb_1[3]) / 2
    bb2_cent = (bb_2[0] + bb_2[2]) / 2, (bb_2[1] + bb_2[3]) / 2

    # Convert quaternions to rotation matrices
    rrot1 = Rotation.from_quat(robot_rot_1).as_matrix()
    rrot2 = Rotation.from_quat(robot_rot_2).as_matrix()

    # Pixel to Camera coordinates
    # Inverting intrinsics to project pixel to ray
    inv_intrinsics = np.linalg.inv(DOTS_INTRINSICS)

    # Calculate Rays
    # Note: This assumes z=1 for the normalized image plane
    ray1 = rrot1 @ (inv_intrinsics @ np.array([bb1_cent[0], bb1_cent[1], 1]))
    ray2 = rrot2 @ (inv_intrinsics @ np.array([bb2_cent[0], bb2_cent[1], 1]))

    # Normalize rays
    ray1 = ray1 / np.linalg.norm(ray1)
    ray2 = ray2 / np.linalg.norm(ray2)

    # Least squares triangulation
    # Solving A * [t1, t2]^T = b
    A = np.vstack([ray1, -ray2]).T
    b = robot_pos_2 - robot_pos_1

    ts, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Calculate points on each ray
    pos1_closest = robot_pos_1 + ts[0] * ray1
    pos2_closest = robot_pos_2 + ts[1] * ray2

    # Midpoint of the shortest segment connecting the two rays
    return (pos1_closest + pos2_closest) / 2


def setup_save_dir(
        output_path: Path,
        task_run: str,
        pipelines: List[List[str]],
        endpoints: EndpointMap,
        test_prompts: Dict,
        date_override: Optional[str] = None
) -> Path:
    """Creates the directory structure for saving experiment results."""
    date = time.strftime("%Y%m%d") if not date_override else date_override
    exp_path = output_path / date / 'real' / task_run

    robots = test_prompts['system']['robots']

    for ex in pipelines:
        ex_dir = '_'.join(ex)
        if ex_dir == "vlm":
            for ep in endpoints["vlm_models"]:
                for robot in robots:
                    path = exp_path / ex_dir / f'{ep.endpoint}_{ep.model_variant}' / robot
                    path.mkdir(parents=True, exist_ok=True)
        else:
            for ep in endpoints["vlm_models"]:
                for lm_ep in endpoints["lm_models"]:
                    for robot in robots:
                        path = exp_path / ex_dir / f'{ep.endpoint}_{ep.model_variant}' / \
                               f'{lm_ep.endpoint}_{lm_ep.model_variant}' / robot
                        path.mkdir(parents=True, exist_ok=True)

    return exp_path


def initialize_endpoints(
        config: Dict,
        prompts: Dict,
        keys_path: Path
) -> EndpointMap:
    """
    Initializes and validates all model endpoints defined in the config.
    """
    endpoints: EndpointMap = {
        "vlm_models": [],
        "lm_models": []
    }

    # Helper for prefixes
    def get_sys_prompt(m_class: str) -> str:
        return prompts["system"]['vlm_prefix'] if m_class == "vlm_models" else prompts["system"]['lm_prefix']

    # --- GPT Models ---
    gpt_conf = config['models']['gpt_models']
    if gpt_conf['endpoints']:
        valid_gpt = {'gpt-4o', 'gpt-4.1', 'gpt-4.5', 'gpt-5', 'o1-preview'}
        for endpoint in gpt_conf['endpoints']:
            if endpoint != "open_ai":
                raise ValueError(f"Unknown GPT endpoint: {endpoint}")

            for model_class in ["lm_models", "vlm_models"]:
                for model in gpt_conf.get(model_class, []):
                    if model not in valid_gpt:
                        raise ValueError(f"Invalid GPT model: {model}")
                    endpoints[model_class].append(GPT(
                        model_variant=model,
                        key_file_path=str(keys_path),
                        system_prompt=get_sys_prompt(model_class)
                    ))

    # --- Gemini Models ---
    gem_conf = config['models']['gemini_models']
    if gem_conf['endpoints']:
        valid_gemini = {"gemini-2.5-pro", "gemma3:12b", "gemini-2.5-pro-preview-03-25",
                        "gemini-2.0-flash", "gemini-2.0-flash-lite"}
        for model_class in ["lm_models", "vlm_models"]:
            for model in gem_conf.get(model_class, []):
                if model not in valid_gemini:
                    raise ValueError(f"Invalid Gemini model: {model}")
                endpoints[model_class].append(Gemini(
                    model_variant=model,
                    key_file_path=str(keys_path),
                    system_prompt=get_sys_prompt(model_class),
                    get_bounding_boxes=True
                ))

    # --- Anthropic Models ---
    anth_conf = config['models']['anthropic_models']
    if anth_conf['endpoints']:
        valid_anth = {"claude-opus-4-20250514", 'claude-opus-4-1-20250805'}
        for model_class in ["lm_models", "vlm_models"]:
            for model in anth_conf.get(model_class, []):
                if model not in valid_anth:
                    raise ValueError(f"Invalid Anthropic model: {model}")
                endpoints[model_class].append(Claude(
                    model_variant=model,
                    key_file_path=str(keys_path),
                    system_prompt=get_sys_prompt(model_class)
                ))

    return endpoints


def extract_bbs_with_grounding_dino(
        affordance: Any,
        save_dir: Path,
        model_name: str,
        img_path: Path,
        log: Log,
        dino_model: Any,
        media_conf: Dict,
        odom_lookup: Optional[OdometryLookup]
) -> List[Any]:
    """Extracts bounding boxes, annotates image, and optionally estimates localization."""
    start = time.time()
    annotated_image, boxes, object_list = grounding_dino_bbs(dino_model, affordance, img_path)
    log.append(f"Grounding DINO took {time.time() - start:.2f} seconds to produce bounding boxes")

    # Save Image
    image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(save_dir / f"GD_{model_name}_{img_path.name}"), image_bgr)

    if media_conf.get("show_images"):
        cv2.imshow("Bounding Boxes", image_bgr)

    # Synthetic from gpt is image only, no trajectory for triangulation
    if 'gpt' in str(img_path):
        objects = [ObjectMap(name=obj, frames=[], estimate=None) for obj in object_list]
        return objects

    # Real data: Use adjacent frames for triangulation
    frame_offset = 20 if 'real' in str(save_dir) else 6
    match = re.search(r"(\d+)", img_path.name)
    if not match:
        raise ValueError(f"Could not extract frame ID from {img_path.name}")
    frame_id = int(match.group(1))

    frames = [LabelledFrame(
        frame_id=frame_id,
        annotated_image=annotated_image,
        bounding_boxes=boxes,
        object_list=object_list
    )]

    # Look for adjacent frames
    parent_dir = img_path.parent
    for file_name in os.listdir(parent_dir):
        match_neighbor = re.search(r"(\d+)", file_name)
        if match_neighbor:
            digits = int(match_neighbor.group(1))
            if digits == frame_id + frame_offset or digits == frame_id - frame_offset:
                a, b, o = grounding_dino_bbs(dino_model, affordance, parent_dir / file_name)
                frames.append(LabelledFrame(
                    frame_id=digits,
                    annotated_image=a,
                    bounding_boxes=b,
                    object_list=o
                ))

    frames = sorted(frames, key=lambda x: x.frame_id)

    # Get configuration for camera ID (assumes media_conf['cam'] exists)
    cam_id_match = re.findall(r'\d+', media_conf.get('cam', '0'))
    cam_id = cam_id_match[0] if cam_id_match else '0'

    return localise_with_nearby_frames(object_list, frames, odom_lookup, cam_id)


def localise_with_nearby_frames(
        object_list: List[str],
        frames: List[Any],
        odom_lookup: Optional[OdometryLookup],
        cam_id: str
) -> List[Any]:
    """Matches objects across frames and prepares them for triangulation."""
    objects = []

    for obj in object_list:
        om = ObjectMap(name=obj, frames=[], estimate=None)

        for frame in frames:
            if om.name in frame.object_list:
                obj_index = frame.object_list.index(obj)

                # Fetch odometry if available
                odometry = None
                if odom_lookup and cam_id in odom_lookup and frame.frame_id in odom_lookup[cam_id]:
                    odometry = odom_lookup[cam_id][frame.frame_id]

                om.frames.append(ObjectLabelledFrame(
                    frame_id=frame.frame_id,
                    annotated_image=frame.annotated_image,
                    bounding_box=frame.bounding_boxes[obj_index],
                    odometry=odometry,
                ))
        objects.append(om)

    if odom_lookup is not None:
        return estimate_object_localisation(objects)
    return objects


def estimate_object_localisation(objects: List[Any]) -> List[Any]:
    """Performs triangulation on the collected object frames."""
    for obj in objects:
        estimates = []
        for i in range(len(obj.frames) - 1):
            frame1 = obj.frames[i]
            frame2 = obj.frames[i + 1]

            if not (frame1.odometry and frame2.odometry):
                continue

            est = triangulate_position(
                robot_pos_1=frame1.odometry.pose.position.to_numpy()[:3],
                robot_pos_2=frame2.odometry.pose.position.to_numpy()[:3],
                robot_rot_1=frame1.odometry.pose.orientation.to_numpy(),
                robot_rot_2=frame2.odometry.pose.orientation.to_numpy(),
                bb_1=frame1.bounding_box,
                bb_2=frame2.bounding_box
            )
            estimates.append(est)

        # Average successful estimates
        valid_estimates = [e for e in estimates if e is not None]
        if valid_estimates:
            obj.estimate = np.mean(np.stack(valid_estimates), axis=0)

    return objects


def vlm_only_pipeline(
        img_path: Path,
        robot: str,
        pipeline_dir: Path,
        system_prompt: str,
        user_prompt: str,
        prompt_id: str,
        log: Log,
        endpoints: EndpointMap,
        dino_model: Any,
        media_conf: Dict,
        odom_lookup: Optional[OdometryLookup]
) -> List[AffordanceResponse]:
    """Runs a VLM-only pipeline to detect affordances."""
    vlm_detections = []

    for endpoint in endpoints['vlm_models']:
        log.append(f"VLM Endpoint: {endpoint.model_class} {endpoint.model_variant}")
        log.append(f"Prompt ID: {prompt_id}", tab_after=True)
        save_dir = pipeline_dir / f'{endpoint.endpoint}_{endpoint.model_variant}' / robot

        start = time.time()
        try:
            affordance = endpoint.get_model_response(
                system_prompt=system_prompt,
                message=user_prompt,
                image_path=img_path
            )

            annotate_affordance_on_image(img_path, save_dir, affordance, endpoint.model_variant,
                                         media_conf.get('show_images', False))
            log.append(f"Took {time.time() - start:.2f} seconds to respond for image: {img_path}")
            log.append(f"Structured Output: {affordance}")

            localisation = None
            if affordance.objects is not None:
                try:
                    localisation = extract_bbs_with_grounding_dino(
                        affordance, save_dir, endpoint.model_variant, img_path,
                        log, dino_model, media_conf, odom_lookup
                    )
                except Exception as e:
                    log.append(f"Failed Object Localisation: {e}")

            vlm_detections.append(AffordanceResponse(
                endpoint=endpoint.endpoint,
                model=f"{endpoint.model_class} {endpoint.model_variant}",
                affordance=affordance,
                object_localisation=localisation
            ))

        except Exception as e:
            log.append(f"Failed Affordance classification: {e}")
            vlm_detections.append(AffordanceResponse(
                endpoint=endpoint.endpoint,
                model=f"{endpoint.model_class} {endpoint.model_variant}",
                affordance=None,
                object_localisation=None
            ))

        log.untab_cursor(1)

    log.append("-------------------------")
    return vlm_detections


def detect_and_lm_pipeline(
        img_path: Path,
        pipeline_dir: Path,
        robot: str,
        vlm_system_prompt: str,
        vlm_user_prompt: str,
        lm_sys_prompt: str,
        lm_usr_prompt: str,
        prompt_id: str,
        log: Log,
        endpoints: EndpointMap,
        dino_model: Any,
        media_conf: Dict,
        odom_lookup: Optional[OdometryLookup]
) -> List[VisionDetection]:
    """Runs a VLM -> LLM pipeline. VLM detects objects, LLM infers affordances."""
    vlm_detections = []

    for endpoint in endpoints['vlm_models']:
        log.append(f"VLM Endpoint: {endpoint.endpoint} {endpoint.model_variant}")
        log.append(f"Prompt ID: {prompt_id}", tab_after=True)
        vlm_dir = pipeline_dir / f"{endpoint.endpoint}_{endpoint.model_variant}"

        start = time.time()
        try:
            response = endpoint.get_model_response(
                system_prompt=vlm_system_prompt,
                message=vlm_user_prompt,
                image_path=img_path
            )
            log.append(f"Took {time.time() - start:.2f} seconds to respond for image: {img_path}")
            log.append(f"Structured Output: {response}")

            # Chained LLM call
            lm_responses = query_lm_with_object_list(
                vlm_dir, robot, img_path, response.objects,
                lm_sys_prompt, lm_usr_prompt, log, endpoints,
                dino_model, media_conf, odom_lookup
            )

            vlm_detections.append(VisionDetection(
                endpoint=endpoint.endpoint,
                model=f"{endpoint.model_class} {endpoint.model_variant}",
                objects=response.objects,
                lm_response=lm_responses
            ))

        except Exception as e:
            log.append(f"Failed: {e}")

        log.untab_cursor(1)

    log.append("-------------------------")
    return vlm_detections


def query_lm_with_object_list(
        vlm_dir: Path,
        robot: str,
        img_path: Path,
        objects: List[str],
        system_prompt: str,
        user_prompt: str,
        log: Log,
        endpoints: EndpointMap,
        dino_model: Any,
        media_conf: Dict,
        odom_lookup: Optional[OdometryLookup]
) -> List[AffordanceResponse]:
    """Queries Language Models with the list of detected objects."""
    # Update LM prompt with detected objects.
    object_str = f"[{','.join(objects)}]"
    augmented_user_prompt = f"I have detected the following objects in my environment: {object_str}\n{user_prompt}"

    responses = []
    log.append(f"Objects passed to LM: {object_str}", tab_after=True)

    for lm_endpoint in endpoints['lm_models']:
        log.append(f"LM Endpoint: {lm_endpoint.model_class} {lm_endpoint.model_variant}...", tab_after=True)
        lm_dir = vlm_dir / f'{lm_endpoint.endpoint}_{lm_endpoint.model_variant}' / robot

        start = time.time()
        try:
            affordance = lm_endpoint.get_model_response(
                message=augmented_user_prompt,
                system_prompt=system_prompt
            )

            annotate_affordance_on_image(
                img_path=img_path, save_dir=lm_dir,
                affordance=affordance, model_name=lm_endpoint.model_variant,
                show_images=media_conf.get("show_images", False)
            )

            log.append(f"Took {time.time() - start:.2f} seconds to respond for image: {img_path}")
            log.append(f"Structured Output: {affordance}")

            localisation = None
            if affordance.objects is not None:
                localisation = extract_bbs_with_grounding_dino(
                    affordance, lm_dir, lm_endpoint.model_variant, img_path,
                    log, dino_model, media_conf, odom_lookup
                )

            responses.append(AffordanceResponse(
                endpoint=lm_endpoint.endpoint,
                model=f"{lm_endpoint.model_class} {lm_endpoint.model_variant}",
                affordance=affordance,
                object_localisation=localisation
            ))

        except Exception as e:
            log.append(f"Failed: {e}")

        log.untab_cursor(1)

    log.untab_cursor(1)
    return responses


def map_objects(frames: List[Any], log: Log) -> Dict[str, Dict[str, List[ObjectWorldData]]]:
    """
    Constructs a 'World Map' by aggregating object detections across frames.
    """
    world_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Cycle through all processed frames
    for frame_it in frames:
        # Use experiment iteration 0 for the world map.
        frame = frame_it.frames[0]
        for pr in frame.prompt_responses:
            for endpoint in pr.affordances:
                # determine model signature
                model_sig = endpoint.model
                responses = endpoint.lm_response if '_lm' in frame.pipeline else [endpoint]

                for response in responses:
                    if '_lm' in frame.pipeline:
                        model_sig += f" {response.model}"

                    affordances = response.affordance
                    if affordances is None:
                        continue

                    # Filter for actual affordance fields (excluding 'objects')
                    aff_fields = [x for x in affordances.model_fields_set if x != 'objects']

                    for aff_field in aff_fields:
                        aff_data_list = getattr(affordances, aff_field)
                        if not aff_data_list:
                            continue

                        for aff_instance in aff_data_list:
                            if not aff_instance.objects:
                                continue

                            for obj_name in aff_instance.objects:
                                # Find localisation estimate for this object
                                if not response.object_localisation:
                                    continue

                                mapped_objs = [x for x in response.object_localisation if x.name == obj_name]
                                if mapped_objs:
                                    pos = mapped_objs[0].estimate
                                    if pos is not None:
                                        # Check if we already have this object at this location (basic clustering)
                                        existing_entry = None
                                        for entry in world_map[pr.prompt_id][model_sig][obj_name]:
                                            if np.linalg.norm(entry['location'] - pos) < 0.1:
                                                existing_entry = entry
                                                break

                                        if existing_entry:
                                            existing_entry['affordances'].append(aff_instance)
                                        else:
                                            world_map[pr.prompt_id][model_sig][obj_name].append({
                                                "location": pos,
                                                "affordances": [aff_instance]
                                            })

    # Convert to final structure
    final_map = {}
    for prompt_id, endpoints_data in world_map.items():
        final_map[prompt_id] = {}
        for ep, objects_data in endpoints_data.items():
            # (Optional: Logic to print similar locations was here in original)

            final_map[prompt_id][ep] = [
                ObjectWorldData(
                    name=obj_name,
                    location=data_list[0]['location'],
                    affordances=data_list[0]['affordances']
                )
                for obj_name, data_list in objects_data.items()
            ]

    return final_map


def launch_experiments(
        log: Log,
        test_frames: List[str],
        sample_rate: int,
        experiment_save_dir: Path,
        image_dir: Path,
        config: Dict,
        endpoints: EndpointMap,
        test_prompts: Dict,
        dino_model: Any,
        odom_lookup: Optional[OdometryLookup] = None,
        prev_experiment: Optional[Dict] = None,
        completed_experiments: Optional[List] = None
) -> None:
    """Main loop to run experiments across frames."""
    experiments = prev_experiment if prev_experiment else {}
    media_conf = config['media']

    for experiment_pipe in config['experiments']['pipelines']:
        done = False
        exp_name = "_".join(experiment_pipe)

        # Check completion status
        if completed_experiments:
            for comp_exp in completed_experiments:
                if comp_exp[0] == experiment_save_dir.parent.name and experiment_save_dir.name in comp_exp[1]:
                    done = True
                    break

        if done or (experiment_save_dir / 'all_results.pkl').exists():
            continue

        # Load partial progress
        frames_path = experiment_save_dir / f'{exp_name}_frames.pkl'
        if frames_path.exists():
            with open(frames_path, 'rb') as f:
                frames = pickle.load(f)
        else:
            frames = []

        experiments[exp_name] = {}
        task = experiment_save_dir.parent.name
        print(f"Running pipeline: {experiment_pipe} in {task}/{experiment_save_dir.name}..")
        log.append(f"Running pipeline: {experiment_pipe}..", tab_after=True)

        pipeline_dir = experiment_save_dir / exp_name

        # Determine task prompt
        task_prompt = ""
        if config['experiments']['inc_task']:
            task_key = 'household_items' if task == 'veo' else task
            task_prompt = test_prompts['system']['tasks'].get(task_key, "")

        iterations = config['experiments'].get('iter', 1)

        for im_name in test_frames:
            match = re.search(r'(\d+)', im_name)
            if not match:
                continue
            num = int(match.group(1))

            # Skip if already processed
            if len(frames) > num // sample_rate:
                continue

            iters = []
            log.append(f"{im_name}", tab_after=True)

            for i in range(iterations):
                print(f"    Processing iteration: {i} of {im_name} at {time.strftime('%H:%M')}")
                log.append(f"Iteration: {i}", tab_after=True)

                img_path = image_dir / im_name
                prompt_responses = []

                robots = test_prompts['system']['robots']

                # --- Pipeline Logic ---
                if "vlm" in experiment_pipe and "lm" in experiment_pipe:
                    # Combined VLM + LM Pipeline
                    for robot, robot_desc in robots.items():
                        prefix = test_prompts['system']['vlm_prefix']
                        robot_part = (test_prompts['system'][
                                          'robot_prefix'] + robot_desc) if robot != 'humanoid' else robot_desc

                        vlm_sys = prefix + robot_part + task_prompt
                        lm_sys = test_prompts['system']['lm_prefix'] + robot_part + task_prompt

                        for vlm_key, vlm_usr in test_prompts['user']['vlm_prompts'].items():
                            for lm_key, lm_usr in test_prompts['user']['lm_prompts'].items():
                                p_id = f"{robot}_{vlm_key}_{robot}_{lm_key}"

                                prompt_responses.append(PromptResponse(
                                    prompt_id=p_id,
                                    system_prompt=[robot, robot],
                                    user_prompt=[vlm_key, lm_key],
                                    affordances=detect_and_lm_pipeline(
                                        img_path, pipeline_dir, robot, vlm_sys, vlm_usr,
                                        lm_sys, lm_usr, p_id, log, endpoints, dino_model,
                                        media_conf, odom_lookup
                                    )
                                ))
                else:
                    # VLM Only Pipeline
                    for robot, robot_desc in robots.items():
                        prefix = test_prompts['system']['vlm_prefix']
                        robot_part = (test_prompts['system'][
                                          'robot_prefix'] + robot_desc) if robot != 'humanoid' else robot_desc
                        sys_prompt = prefix + robot_part + task_prompt

                        for up_key, usr_prompt in test_prompts['user']['vlm_prompts'].items():
                            p_id = f"{robot}_{up_key}_None_None"
                            prompt_responses.append(PromptResponse(
                                prompt_id=p_id,
                                system_prompt=[robot, None],
                                user_prompt=[up_key, None],
                                affordances=vlm_only_pipeline(
                                    img_path, robot, pipeline_dir, sys_prompt, usr_prompt,
                                    p_id, log, endpoints, dino_model, media_conf, odom_lookup
                                )
                            ))

                iters.append(Frame(
                    pipeline=exp_name,
                    frame_num=num,
                    frame=np.array(Image.open(img_path).convert('RGB')),
                    prompt_responses=prompt_responses,
                ))
                log.untab_cursor(1)  # End iteration loop

            frames.append(FrameIteration(frames=iters, frame_num=num))

            # Save checkpoint
            with open(experiment_save_dir / f'{exp_name}_frames.pkl', 'wb') as f:
                pickle.dump(frames, f)

        log.untab_cursor(1)  # End frames loop

        world_map = None
        if config['experiments'].get('build_world_map'):
            print("Building world map")
            world_map = map_objects(frames, log)

        experiment_data = Experiment(frames=frames, world_map=world_map)

        # Save results
        with open(experiment_save_dir / f'{exp_name}_results.pkl', 'wb') as f:
            pickle.dump(experiment_data, f)

        experiments[exp_name] = experiment_data
        with open(experiment_save_dir / 'all_results.pkl', 'wb') as f:
            pickle.dump(experiments, f)

        log.append("--------------------------------\n")


def run_real_experiments(
        config: Dict,
        endpoints: EndpointMap,
        test_prompts: Dict,
        dino_model: Any,
        save_root: Path
) -> None:
    """Runner for real-world data experiments."""
    media_conf = config['media']
    media_root = Path(ROOT, 'datasets', 'real_data', 'experiments', media_conf['date'])

    tasks = media_conf.get('tasks', [])
    for task in tasks:
        runs = os.listdir(media_root / task) if 'all' in media_conf['runs'] else media_conf['runs']
        runs = [r for r in runs if not r.startswith('.')]  # Filter hidden files

        for run in runs:
            experiment_save_dir = setup_save_dir(
                save_root / 'experiments', f'{task}/{run}',
                config['experiments']['pipelines'], endpoints, test_prompts,
                config['experiments'].get('date_override')
            )

            log = Log(experiment_save_dir)
            log.append(f"Sampling task: {task} \nRun: {run}", tab_after=True)

            media_dir = media_root / task / run
            image_dir = media_dir / media_conf['cam'] / 'frames'

            ims = sorted(os.listdir(image_dir))
            test_frames = ims[::media_conf['sample_rate']]

            odom_path = media_dir / "odometry_lookup.pkl"
            odom_lookup = pickle.load(open(odom_path, 'rb')) if odom_path.exists() else None

            # Example of skipped experiments logic (could be externalized)
            completed_experiments = []

            launch_experiments(
                log, test_frames, media_conf['sample_rate'], experiment_save_dir,
                image_dir, config, endpoints, test_prompts, dino_model,
                odom_lookup=odom_lookup, completed_experiments=completed_experiments
            )


def extract_number(filename: str) -> int:
    """Helper to sort filenames numerically."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def run_synthetic_experiments(
        config: Dict,
        endpoints: EndpointMap,
        test_prompts: Dict,
        dino_model: Any,
        save_root: Path
) -> None:
    """Runner for synthetic data experiments."""
    media_conf = config['media']

    for synth_src in media_conf['synth_source']:
        media_root = Path(ROOT, 'datasets', 'synthetic_data', synth_src)

        if synth_src == 'gpt':
            sample_rate = 1
            media_root = media_root / "images"

            # Iterate tasks
            for task_dir in [x for x in media_root.iterdir() if x.is_dir()]:
                # Iterate image sets
                for im_dir in [x for x in task_dir.iterdir() if x.is_dir()]:

                    experiment_save_dir = setup_save_dir(
                        save_root / 'experiments',
                        f'synthetic/{synth_src}/{task_dir.name}/{im_dir.name}',
                        config['experiments']['pipelines'], endpoints, test_prompts,
                        config['experiments'].get('date_override')
                    )

                    if (experiment_save_dir / 'all_results.pkl').exists():
                        continue

                    log = Log(experiment_save_dir)
                    test_frames = sorted([x.name for x in im_dir.iterdir() if x.suffix == '.png'], key=extract_number)

                    launch_experiments(
                        log, test_frames, sample_rate, experiment_save_dir,
                        im_dir, config, endpoints, test_prompts, dino_model
                    )

        elif synth_src == 'veo':
            sample_rate = 12
            # Iterate video dirs
            for im_dir in [x for x in media_root.iterdir() if x.is_dir() and 'video' in x.name]:
                experiment_save_dir = setup_save_dir(
                    save_root / 'experiments',
                    f'synthetic/{synth_src}/{im_dir.name}',
                    config['experiments']['pipelines'], endpoints, test_prompts
                )

                log = Log(experiment_save_dir)
                test_frames = sorted([x.name for x in im_dir.iterdir() if x.suffix == '.png'])[::sample_rate]

                launch_experiments(
                    log, test_frames, sample_rate, experiment_save_dir,
                    im_dir, config, endpoints, test_prompts, dino_model
                )


if __name__ == '__main__':
    # Load Configurations and Resources
    config_data = load_config(config_path=ROOT / "configs" / "experiment_config.yaml")
    
    # Setup Environment
    setup_environment(config_data)

    if config_data['paths']['save_root']:
        save_root = Path(config_data['paths']['save_root'])
    else:
        save_root = ROOT

    with open(ROOT / 'prompts' / 'prompts.json', 'r') as f:
        prompts_data = json.load(f)

    grounding_dino_model = load_dino_model(config_data)

    # 4. Initialize Models
    endpoint_map = initialize_endpoints(config_data, prompts_data, ROOT / 'keys.json')

    # 5. Run Experiments
    media_sources = config_data['media']['source_material']

    for source in media_sources:
        if source == "real":
            run_real_experiments(config_data, endpoint_map, prompts_data, grounding_dino_model, save_root)
        else:
            run_synthetic_experiments(config_data, endpoint_map, prompts_data, grounding_dino_model, save_root)