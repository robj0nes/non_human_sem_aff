import math
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import supervision as sv
import torch
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sentence_transformers import SentenceTransformer, util
from torchvision.ops import box_convert

# Third-party imports
from groundingdino.util.inference import load_image, predict

# Constants
EPSILON = 1e-6
logger = logging.getLogger(__name__)


def semantic_sim_matching(annotations: List[str], predictions: List[str]) -> torch.Tensor:
    """
    Computes semantic similarity scores between lists of annotation strings and prediction strings.

    Args:
        annotations: List of ground truth label strings.
        predictions: List of predicted label strings.

    Returns:
        A tensor of shape (N, M) containing cosine similarity scores.
    """
    if not annotations or not predictions:
        return torch.tensor([])

    try:
        sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb1 = sim_model.encode(annotations, convert_to_tensor=True)
        emb2 = sim_model.encode(predictions, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2)
    except Exception as e:
        logger.error(f"Error in semantic similarity matching: {e}")
        return torch.tensor([])


def evaluate(data: Dict[str, Any], labels: Dict[str, Any], sim_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Matches pipeline predictions to ground truth labels based on semantic similarity.

    Args:
        data: The experimental results data structure (assumed custom object with dot notation).
        labels: Dictionary of ground truth annotations keyed by frame number.
        sim_threshold: Threshold for cosine similarity to consider a text match valid.

    Returns:
        A nested dictionary containing evaluation matches and metadata.
    """
    evals = {}

    for pipeline_name, pipeline_data in data.items():
        evals[pipeline_name] = {}

        # Iterate through frames in the pipeline data
        for frame in pipeline_data.frames:
            frame_num = frame.frame_num

            if frame_num not in labels:
                continue

            evals[pipeline_name][frame_num] = {}
            label_entry = labels[frame_num]

            # Iterate through iterations (repeats of the experiment)
            for i, frame_iteration in enumerate(frame.frames):
                evals[pipeline_name][frame_num][i] = {}

                for response in frame_iteration.prompt_responses:
                    # Parse robot name from prompt ID (e.g. "humanoid_01" -> "Humanoid")
                    robot = response.prompt_id.split('_')[0].title()
                    evals[pipeline_name][frame_num][i][robot] = {}

                    # Copy and normalise annotations
                    annotation = label_entry[robot].copy() if robot in label_entry else {}
                    if robot == "Humanoid":
                        if 'Pick' in annotation:
                            annotation['Grasp or Pick'] = annotation.pop('Pick')
                        if 'Grasp' in annotation:
                            annotation['Grasp or Pick'] = annotation.pop('Grasp')

                    labelled_objects = label_entry.get('objects', [])

                    # Process each endpoint (affordance prediction)
                    for endpoint in response.affordances:
                        affordance_map = endpoint.affordance

                        # Match affordances
                        affordance_matches = []
                        if affordance_map and affordance_map.affordances:
                            labelled_affs = list(annotation.keys())
                            predicted_affs = [x.name for x in affordance_map.affordances]

                            if labelled_affs and predicted_affs:
                                aff_scores = semantic_sim_matching(labelled_affs, predicted_affs)

                                # Find indices where similarity > threshold
                                match_indices = torch.nonzero(aff_scores > sim_threshold, as_tuple=False)

                                for idx in match_indices:
                                    lbl_idx, pred_idx = idx[0].item(), idx[1].item()
                                    match_entry = {
                                        "aff": labelled_affs[lbl_idx],
                                        "response": affordance_map.affordances[pred_idx],
                                    }

                                    # Match objects within the matched affordance
                                    model_detections = match_entry['response'].objects
                                    if labelled_objects and model_detections:
                                        obj_scores = semantic_sim_matching(labelled_objects, model_detections)
                                        # Check if any object matches satisfy threshold
                                        similarity_match = \
                                        torch.any(obj_scores >= sim_threshold, dim=-1).nonzero(as_tuple=True)[0]
                                        match_entry['obj_matches'] = [
                                            labelled_objects[x] for x in similarity_match if x < len(labelled_objects)
                                        ]
                                    else:
                                        match_entry['obj_matches'] = []

                                    affordance_matches.append(match_entry)

                        evals[pipeline_name][frame_num][i][robot][endpoint.endpoint] = {
                            "model": endpoint.model,
                            "labels": annotation,
                            "matches": affordance_matches
                        }
    return evals


def gd_annotate_override(image_source: np.ndarray, boxes: torch.Tensor,
                         logits: torch.Tensor, phrases: List[str], class_id: np.ndarray) -> np.ndarray:
    """
    Annotates an image with bounding boxes and labels using Supervision.
    """
    h, w, _ = image_source.shape
    # Scale normalized boxes to pixel coordinates
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    detections = sv.Detections(xyxy=xyxy, class_id=class_id)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS)

    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame


def grounding_dino_debug(dino_model: Any, objects: Dict[str, Any], image_path: Union[str, Path],
                         robot: str, debug_dir: Union[str, Path]) -> None:
    """
    Runs GroundingDINO inference for debugging purposes and saves the annotated image.
    """
    try:
        image_source, image = load_image(str(image_path))

        # Construct prompt from object list
        prompt = ' . '.join(obj for obj in objects['objects']) + '.'

        # Inference
        boxes, logits, object_list = predict(
            model=dino_model,
            image=image,
            caption=prompt,
            box_threshold=0.35,
            text_threshold=0.25,
            device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Assign class IDs based on TP/FP/FN/TN status passed in `objects`
        class_ids = []
        for obj in object_list:
            obj_title = obj.title()
            if obj_title in objects.get('TP', set()):
                class_ids.append(0)
            elif obj_title in objects.get('FP', set()):
                class_ids.append(1)
            elif obj_title in objects.get('FN', set()):
                class_ids.append(2)
            elif obj_title in objects.get('TN', set()):
                class_ids.append(3)
            else:
                class_ids.append(4)

        annotated_image = gd_annotate_override(
            image_source=image_source, boxes=boxes, logits=logits,
            phrases=object_list, class_id=np.array(class_ids, dtype=int)
        )

        # Add text overlays
        cv2.putText(annotated_image, f"{robot}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        text_start_y = 60
        offset = 40
        for i, key in enumerate(objects.keys()):
            if key != 'objects':
                text = f"{key}: {list(objects[key])}"
                cv2.putText(annotated_image, text, (50, text_start_y + offset * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # Save result
        save_path = Path(debug_dir) / f"{robot}_{Path(image_path).name}"
        cv2.imwrite(str(save_path), annotated_image)

    except Exception as e:
        logger.error(f"Failed to run GroundingDINO debug for {image_path}: {e}")


def _calculate_classification_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Helper to calculate standard classification metrics."""
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = 2 * (precision * recall) / (precision + recall + EPSILON)

    accuracy = (tn + tp) / (tn + tp + fn + fp + EPSILON)
    balanced_accuracy = 0.5 * ((tp / (tp + fn + EPSILON)) + (tn / (tn + fp + EPSILON)))

    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + EPSILON
    mcc = (tp * tn - fp * fn) / mcc_denom

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'mcc': mcc
    }


def calculate_match_score(test_data: Dict, label_data: Dict, frame_dir: Union[str, Path],
                          real: bool = True, name: Optional[str] = None,
                          dino_model: Any = None, debug_mode: bool = False) -> Dict:
    """
    Calculates True Positives, False Positives, False Negatives, True Negatives and derived metrics.
    """
    scores = {
        'openai': {},
        'open_api': {},  # Gemeni usually maps to this key in this codebase
        'anthropic': {}
    }

    frame_dir = Path(frame_dir)

    # 1. Calculate Raw Counts (TP, FP, FN, TN)
    for p_key in test_data:
        for frame_num in test_data[p_key]:
            for iter_idx in test_data[p_key][frame_num]:
                for robot in test_data[p_key][frame_num][iter_idx]:
                    for model in test_data[p_key][frame_num][iter_idx][robot]:

                        # Initialize structure if missing
                        if robot not in scores[model]:
                            scores[model][robot] = {'frames': {}}
                        if frame_num not in scores[model][robot]['frames']:
                            scores[model][robot]['frames'][frame_num] = []

                        matches = test_data[p_key][frame_num][iter_idx][robot][model].get("matches", [])
                        aff_scores = {}

                        # Get Ground Truth keys
                        gt_affordances = label_data[frame_num][robot].keys()

                        for affordance_label in gt_affordances:
                            aff_obj_labels = label_data[frame_num][robot][affordance_label]

                            # Normalize Humanoid labels
                            norm_label = affordance_label
                            if robot == "Humanoid" and affordance_label in ['Pick', 'Grasp']:
                                norm_label = 'Grasp or Pick'

                            aff_scores[norm_label] = {
                                'affordance': norm_label,
                                'labelled_objects': aff_obj_labels,
                                'real_data': real,
                                'all_objects': label_data[frame_num]['objects'],
                                'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                                'tps': set(), 'fps': set(), 'fns': set(), 'tns': set()
                            }

                            current_score = aff_scores[norm_label]
                            matched = False

                            for match in matches:
                                if match['aff'] == norm_label:
                                    matched = True
                                    predicted_objs = set(match['obj_matches'])
                                    ground_truth_objs = set(aff_obj_labels)
                                    all_scene_objs = set(current_score['all_objects'])

                                    # Set Operations
                                    current_score['tps'] = predicted_objs & ground_truth_objs
                                    current_score['fps'] = predicted_objs - ground_truth_objs
                                    current_score['fns'] = ground_truth_objs - predicted_objs
                                    current_score['tns'] = all_scene_objs - (
                                                current_score['tps'] | current_score['fps'] | current_score['fns'])

                                    # Count Operations
                                    current_score['tp'] += len(current_score['tps'])
                                    current_score['fp'] += len(current_score['fps'])
                                    current_score['fn'] += len(current_score['fns'])
                                    current_score['tn'] += len(current_score['tns'])

                                    if debug_mode and dino_model and name:
                                        # Construct Image Path logic
                                        img_name = f'{name.replace("_", " ")}_{frame_num}.png' if 'gpt' in str(
                                            frame_dir) \
                                            else f'frame{frame_num:04}.png' if 'veo' in str(frame_dir) \
                                            else f'frame{frame_num:06}.png'

                                        image_path = frame_dir / img_name

                                        # Deduce debug dir
                                        debug_dir = str(frame_dir).replace('frames', 'debugs').replace('images',
                                                                                                       'debugs')
                                        Path(debug_dir).mkdir(parents=True, exist_ok=True)

                                        # Prepare debug objects dict
                                        debug_objs = {
                                            'objects': list(
                                                current_score['tps'] | current_score['fps'] | current_score['fns']),
                                            'TP': current_score['tps'],
                                            'FP': current_score['fps'],
                                            'FN': current_score['fns'],
                                            'TN': current_score['tns']
                                        }
                                        grounding_dino_debug(dino_model, debug_objs, image_path, robot, debug_dir)

                            if not matched:
                                all_objs = set(current_score['all_objects'])
                                current_score['fp'] = 0
                                current_score['fn'] = len(all_objs)
                                current_score['fns'] = all_objs
                                current_score['tn'] = 0
                                current_score['tp'] = 0

                        scores[model][robot]['frames'][frame_num].append(aff_scores)

    # 2. Aggregate and Compute Metrics (Precision, Recall, etc.)
    for model in scores:
        for robot in scores[model]:

            # Aggregate per Affordance and Object
            for frame_key in scores[model][robot]['frames']:
                for i, iter_data in enumerate(scores[model][robot]['frames'][frame_key]):
                    for aff, aff_data in iter_data.items():

                        if aff not in scores[model][robot]:
                            scores[model][robot][aff] = {}
                        if i not in scores[model][robot][aff]:
                            scores[model][robot][aff][i] = {'objects': {}}

                        # Aggregate per object
                        for obj in aff_data['all_objects']:
                            obj_dict = scores[model][robot][aff][i]['objects']
                            if obj not in obj_dict:
                                obj_dict[obj] = {'count': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

                            target = obj_dict[obj]
                            if obj in aff_data['tps']: target['tp'] += 1
                            if obj in aff_data['fps']: target['fp'] += 1
                            if obj in aff_data['fns']: target['fn'] += 1
                            if obj in aff_data['tns']: target['tn'] += 1
                            target['count'] += 1

            # Compute final metrics per Affordance
            for aff in list(scores[model][robot].keys()):
                if aff == 'frames': continue

                for i in list(scores[model][robot][aff].keys()):
                    iter_score = scores[model][robot][aff][i]

                    # A. Metrics per individual object
                    for obj, obj_stats in iter_score['objects'].items():
                        metrics = _calculate_classification_metrics(
                            obj_stats['tp'], obj_stats['fp'], obj_stats['tn'], obj_stats['fn']
                        )
                        obj_stats.update(metrics)

                    # B. Totals for the affordance (summing across frames)
                    frame_data_list = scores[model][robot]['frames']
                    # We need to sum the tp/fp/fn/tn for this specific iteration 'i' and affordance 'aff' across all frames
                    total_tp = sum(frame_data_list[x][i][aff]['tp'] for x in frame_data_list)
                    total_fp = sum(frame_data_list[x][i][aff]['fp'] for x in frame_data_list)
                    total_fn = sum(frame_data_list[x][i][aff]['fn'] for x in frame_data_list)
                    total_tn = sum(frame_data_list[x][i][aff]['tn'] for x in frame_data_list)

                    iter_score['totals'] = _calculate_classification_metrics(total_tp, total_fp, total_tn, total_fn)

    return scores


def build_scoring_dataframe(score_path: Union[str, Path], pickle_loader: Any = None) -> pd.DataFrame:
    """
    Loads a scores pickle file and flattens it into a Pandas DataFrame.
    """
    path = Path(score_path)
    if not path.exists():
        logger.error(f"Score path does not exist: {path}")
        return pd.DataFrame()

    if pickle_loader is None:
        pickle_loader = pickle

    with open(path, "rb") as f:
        scores = pickle_loader.load(f)

    rows = []
    for model, model_data in scores.items():
        for robot, robot_data in model_data.items():
            for aff, aff_data in robot_data.items():
                if aff == 'frames':
                    continue

                for iteration, iter_data in aff_data.items():
                    for obj_name, obj_stats in iter_data['objects'].items():
                        rows.append({
                            'model': model,
                            'robot': robot,
                            'labelled_objects': obj_name,
                            'affordance': aff,
                            'iteration': iteration,
                            'tp': obj_stats['tp'],
                            'fp': obj_stats['fp'],
                            'tn': obj_stats['tn'],
                            'fn': obj_stats['fn'],
                            'precision': obj_stats['precision'],
                            'recall': obj_stats['recall'],
                            'f1': obj_stats['f1'],
                            'accuracy': obj_stats['accuracy'],
                            'balanced_accuracy': obj_stats['balanced_accuracy'],
                            'mcc': obj_stats['mcc']
                        })

    return pd.DataFrame(rows)


def plot_score_matrix(df: pd.DataFrame, model: str, grouping: str, name: str,
                      suffix: str, plot_dir: Union[str, Path]) -> None:
    """Generates a confusion matrix-style plot for scores."""
    idx = pd.IndexSlice
    plot_dir = Path(plot_dir)

    confusion_data = df.pivot_table(
        index='affordance',
        columns='labelled_objects',
        values=['tp', 'fp', 'tn', 'fn'],
        aggfunc='sum'
    )
    metric_colors = {'tp': '#4CAF50', 'fp': '#F44336', 'tn': '#2196F3', 'fn': '#FF9800'}
    fontsize = 18

    # Calculate Means and Stds
    for metric in ['tp', 'fp', 'tn', 'fn']:
        confusion_data[(metric, 'mean', 'Mean')] = confusion_data.loc[:, idx[metric, 'mean', :]].mean(axis=1)
        confusion_data[(metric, 'std', 'Mean')] = confusion_data.loc[:, idx[metric, 'std', :]].mean(axis=1)

    affordances = confusion_data.index
    objects = list(confusion_data.columns.levels[2])
    if 'Mean' not in objects:
        objects.append('Mean')

    # Setup Plot
    fig, axarr = plt.subplots(len(affordances), len(objects), figsize=(2 * len(objects), 2 * len(affordances)))

    # Normalize axarr to 2D list
    if len(affordances) == 1 and len(objects) == 1:
        axarr = [[axarr]]
    elif len(affordances) == 1:
        axarr = [axarr]
    elif len(objects) == 1:
        axarr = [[ax] for ax in axarr]

    def draw_normalized_cell(ax, values, alpha=None):
        """Helper to draw the 4-quadrant cell."""
        # Handle NaNs
        values_cleaned = {
            "means": {k: (0 if pd.isna(v[0]) else v[0]) for k, v in values.items()},
            "stds": {k: (0 if pd.isna(v[1]) else v[1]) for k, v in values.items()}
        }

        # Normalize
        total = sum(values_cleaned["means"].values())
        norm_values = {}
        if total > 0:
            for k, v in values_cleaned["means"].items():
                norm_values[k] = v / total
        else:
            norm_values = {k: 0 for k in values_cleaned["means"]}

        labels = {k: f"{norm_values[k]:.2f}" for k in norm_values}
        positions = {
            'tp': (0.25, 0.75), 'fp': (0.75, 0.75),
            'fn': (0.25, 0.25), 'tn': (0.75, 0.25)
        }

        for key in positions:
            x, y = positions[key]
            curr_alpha = alpha if alpha is not None else norm_values[key]
            ax.add_patch(patches.Rectangle(
                (x - 0.24, y - 0.24), 0.48, 0.48,
                color=metric_colors[key], alpha=curr_alpha
            ))
            ax.text(x, y, labels[key], ha='center', va='center', fontsize=fontsize, color='black')

    # Draw Logic
    for i, aff in enumerate(affordances):
        for j, obj in enumerate(objects):
            ax = axarr[i][j]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            try:
                vals = {}
                for m in ['tp', 'fp', 'tn', 'fn']:
                    vals[m] = (
                        confusion_data.loc[aff, (m, 'mean', obj)],
                        confusion_data.loc[aff, (m, 'std', obj)]
                    )

                if obj == 'Mean':
                    draw_normalized_cell(ax, vals, alpha=1.0)
                else:
                    draw_normalized_cell(ax, vals)
            except KeyError:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=fontsize, style='italic', color='gray')

            if i == 0:
                ax.set_title(obj, fontsize=fontsize, ha='center', va='center')
            if j == 0:
                ax.text(-0.3, 0.5, aff, va='center', ha='right', fontsize=fontsize, rotation=0)

    plt.tight_layout()

    save_dir = plot_dir / "score_matrix"
    save_dir.mkdir(exist_ok=True)

    model_name = 'GPT' if model == 'openai' else 'Gemini' if model == 'open_api' else 'Claude'
    plt.savefig(save_dir / f"{model_name}_{grouping}_{name}_{suffix}_confusion.pdf")
    plt.close()


def plot_heatmap(fig_x: int, fig_y: int, metric: str, model: str, model_based_df: pd.DataFrame,
                 grouping: str, name: str, plot_dir: Union[str, Path]) -> None:
    """Generates a heatmap for specific metrics."""
    plot_dir = Path(plot_dir)

    heatmap_data = model_based_df.pivot_table(
        index='affordance',
        columns='labelled_objects',
        values=metric,
        aggfunc='mean'
    )

    # Calculate Marginal Means
    heatmap_data[("mean", "Affordance\n Mean")] = heatmap_data['mean'].mean(axis=1)
    heatmap_data[("std", "Affordance\n Mean")] = heatmap_data['std'].mean(axis=1)

    mean_row = heatmap_data.mean(axis=0).to_frame().T
    mean_row.index = ["Object\n Mean"]
    heatmap_data = pd.concat([heatmap_data, mean_row], axis=0)

    mean_hm, std_hm = heatmap_data['mean'], heatmap_data['std']
    annot = mean_hm.round(2).astype(str) + "\n±\n" + std_hm.map(lambda x: f"{x:.2f}")

    plt.figure(figsize=(10, fig_y) if grouping == 'Clustered' else (fig_x, fig_y))

    # Styling
    label_size = 12
    font_size = 14
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": label_size,
        "ytick.labelsize": label_size,
        "legend.fontsize": font_size,
    })

    ax = sns.heatmap(mean_hm, vmin=0, vmax=1, annot=annot, fmt="", cmap="YlGnBu",
                     cbar_kws={'label': f'{metric.title()}'})

    # Draw Borders around heatmap
    nrows, ncols = mean_hm.shape
    ax.add_patch(Rectangle((0, nrows - 1), ncols, 1, fill=False, edgecolor="black", lw=1.5, clip_on=False))
    ax.add_patch(Rectangle((ncols - 1, 0), 1, nrows, fill=False, edgecolor="black", lw=1.5, clip_on=False))

    model_name = 'GPT' if model == 'openai' else 'Gemini' if model == 'open_api' else 'Claude'
    formatted_metric = ' '.join([x.title() for x in metric.split('_')])
    formatted_name = ' '.join([x.title() for x in name.split('_')])

    plt.title(
        f"{model_name} {formatted_metric} For {grouping.title()} Affordance-Object Characterisation\n {formatted_name}")
    plt.xlabel("Object" if grouping != 'Clustered' else "Object Cluster")
    plt.ylabel("Affordance")
    plt.tight_layout()

    plt.savefig(plot_dir / f"{model_name}_{grouping}_{name}_{metric}_heatmap.pdf", bbox_inches='tight')
    plt.close()


def plot_performance_matrix(df: pd.DataFrame, metrics: List[str], plot_dir: Union[str, Path],
                            mappings: Dict[str, Any], name: Optional[str] = None, fig_x: int = 10) -> None:
    """
    Orchestrates the plotting of performance matrices (Heatmaps and Confusion Grids).

    Args:
        df: Input DataFrame containing scores.
        metrics: List of metrics to plot (e.g. ['f1', 'recall']).
        plot_dir: Directory to save plots.
        mappings: Configuration dict for renames and clustering.
        name: Base name for the plot files.
        fig_x: Figure width.
    """
    plot_dir = Path(plot_dir)

    # 1. Apply Renames
    df['affordance'] = df['affordance'].replace(mappings.get('affordance_renames', {}))
    df['labelled_objects'] = df['labelled_objects'].replace(mappings.get('object_renames', {}))

    # 2. Fix known labeling issues (Cuttable objects)
    cuttable_objects = mappings.get('cuttable_objects', [])
    mask = (df['labelled_objects'].isin(cuttable_objects)) & \
           (df['affordance'] == 'Cut') & \
           (df['fp'] > 0)

    if mask.any():
        df.loc[mask, 'tp'] += df.loc[mask, 'fp']
        df.loc[mask, 'fp'] = 0

    # 3. Create Clustered DataFrame
    cluster_map = mappings.get('clusters', {})
    reverse_map = {item: category for category, items in cluster_map.items() for item in items}
    clustered_df = df.copy()
    clustered_df['labelled_objects'] = clustered_df['labelled_objects'].replace(reverse_map)

    metric_cols = ['tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy', 'mcc']

    # 4. Generate Plots
    for data_frame, grouping in zip([df, clustered_df], ["Individual", 'Clustered']):
        # Pre-group by iterations to get means first
        grouped_df = data_frame.groupby(
            ['model', 'robot', 'labelled_objects', 'affordance', 'iteration'], as_index=False
        )[metric_cols].mean()

        for model in grouped_df['model'].unique():
            model_df = grouped_df[grouped_df['model'] == model]

            # Split by Humanoid / Non-Humanoid
            subsets = [
                (model_df[model_df['robot'] == 'Humanoid'], 'Humanoid', 6),
                (model_df[model_df['robot'] != 'Humanoid'], 'Non-Humanoid', 8)
            ]

            for df_subset, suffix, fig_y in subsets:
                if df_subset.empty:
                    continue

                # Aggregate iterations (Mean and Std)
                agg_df = df_subset.groupby(
                    ['model', 'robot', 'affordance', 'labelled_objects']
                )[metric_cols].agg(["mean", "std"])

                for metric in metrics:
                    plot_heatmap(fig_x, fig_y, metric, model, agg_df, grouping, f'{name}_{suffix}', plot_dir)

                # Note: plot_score_matrix uses all metrics internally, so we call it once per subset/grouping
                plot_score_matrix(agg_df, model, grouping, name, suffix, plot_dir)