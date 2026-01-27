import os
import pickle
import pandas as pd
import logging

# local files
import compute
from src.utils import load_dino_model
from utils import load_pickle, save_pickle, load_config

from pathlib import Path
from typing import Dict, Any, Optional, Union

# Optional import if debugging.
from groundingdino.util.inference import load_model

# Constants
ROOT = Path(__file__).resolve().parent.parent

def run_single_evaluation(
        data_path: Path,
        label_path: Path,
        save_path: Path,
        overwrite: bool,
        sim_threshold: float,
        log_name: str
) -> None:
    """
    Helper function to run a single evaluation step: load data/labels, compute metrics, and save.
    """
    if save_path.exists() and not overwrite:
        return

    if data_path.exists() and label_path.exists():
        logging.info(f'Evaluating {log_name}')
        data = load_pickle(data_path)
        labels = load_pickle(label_path)

        # Compute evaluation metrics
        evals = compute.evaluate(data, labels, sim_threshold=sim_threshold)
        save_pickle(evals, save_path)

    elif not data_path.exists():
        # Implemented optional logging note: Warn if expected data is missing
        logging.warning(f"Skipping {log_name}: Data results file not found at {data_path}")

    elif not label_path.exists():
        logging.warning(f"Skipping {log_name}: Label file not found at {label_path}")


def real_world_evals(
        scenario: str,
        run: str,
        res_dir: Path,
        label_dir: Path,
        eval_dir: Path,
        overwrite: bool = False,
        sim_threshold: float = 0.5
) -> None:
    """Evaluates real-world data scenarios."""
    os.makedirs(eval_dir, exist_ok=True)

    save_path = eval_dir / f'{scenario}_{run}_eval.pkl'
    data_path = res_dir / scenario / run / "all_results.pkl"
    label_path = label_dir / f"{scenario}_{run}_cam0_annotations_24fps.pkl"

    run_single_evaluation(
        data_path, label_path, save_path, overwrite, sim_threshold, log_name=f"{scenario}_{run}"
    )


def eval_synth_video(
        data_root: Path,
        label_root: Path,
        save_dir: Path,
        overwrite: bool = False,
        sim_threshold: float = 0.5
) -> None:
    """Evaluates synthetic video data."""
    videos = ['video1', 'video2', 'video3', 'video4']

    for video in videos:
        save_path = save_dir / f'{video}_eval.pkl'
        data_path = data_root / video / "all_results.pkl"
        label_path = label_root / "veo" / "labels" / f"{video}_annotations_12fps.pkl"

        run_single_evaluation(
            data_path, label_path, save_path, overwrite, sim_threshold, log_name=video
        )


def eval_synth_images(
        data_root: Path,
        label_root: Path,
        save_dir: Path,
        overwrite: bool = False,
        sim_threshold: float = 0.5
) -> None:
    """Evaluates synthetic image data (GPT generated)."""
    gpt_labels_dir = label_root / "gpt" / "images"

    if not gpt_labels_dir.exists():
        logging.warning(f"GPT Label directory not found at {gpt_labels_dir}")
        return

    for task in os.listdir(gpt_labels_dir):
        task_data_path = data_root / task

        # Skip files or non-existent directories
        if '.' in task or not task_data_path.exists():
            continue

        task_eval_dir = save_dir / task
        task_eval_dir.mkdir(parents=True, exist_ok=True)

        # Iterate through items in the task directory
        sorted_items = sorted(os.listdir(gpt_labels_dir / task))
        for item_name in sorted_items:
            # Skip metadata folders/files
            if item_name in ['eval_metrics', 'plots', 'scores'] or '.' in item_name:
                continue

            item_data_path = task_data_path / item_name / "all_results.pkl"

            # Note: We let run_single_evaluation handle the existence check and logging
            # so we don't need to double-check item_data_path here, but the loop logic
            # in the original code filtered it early. We keep the check to match logic flow.
            if not item_data_path.exists():
                continue

            save_path = task_eval_dir / f'{item_name}_eval.pkl'
            label_path = gpt_labels_dir / task / item_name / "annotations_1fps.pkl"

            run_single_evaluation(
                item_data_path, label_path, save_path, overwrite, sim_threshold, log_name=item_name
            )


def build_eval_dataframe_recursive(
        dataframe: pd.DataFrame,
        eval_subdir: Path,
        data_dir: Path,
        label_dir: Path,
        synth_source: str,
        score_dir: Path,
        grounding_dino_model: Any,
        debug: bool,
        task: Optional[str] = None
) -> pd.DataFrame:
    """
    Walks through evaluation directories, calculates match scores, and accumulates them into a DataFrame.
    """
    if not eval_subdir.exists():
        logging.warning(f"Evaluation subdirectory not found: {eval_subdir}")
        return dataframe

    for file_name in os.listdir(eval_subdir):
        # Filter for valid .pkl files
        if file_name.startswith('.') or file_name in ['eval_metrics', 'plots', 'scores'] or '.pkl' not in file_name:
            continue

        # Extract name from filename (remove extension and suffix)
        name = file_name.split('.')[0].rsplit('_', 1)[0]
        logging.info(f'Scoring {name}')

        # Determine frame directory based on source
        if synth_source == 'gpt':
            if task is None:
                raise ValueError("Task name is required for GPT synthetic source.")
            frame_dir = data_dir / 'images' / task / name
        else:
            frame_dir = data_dir / name

        if debug:
            debug_path = data_dir / 'debugs'
            if synth_source == 'gpt' and task:
                debug_path = debug_path / task / name

            os.makedirs(debug_path, exist_ok=True)

        # Load predictions and labels
        eval_path = eval_subdir / f"{name}_eval.pkl"
        preds = load_pickle(eval_path)

        if synth_source == 'gpt':
            label_path = label_dir / task / name / "annotations_1fps.pkl"
        else:
            label_path = label_dir / f"{name}_annotations_12fps.pkl"

        labels = load_pickle(label_path)

        # Calculate scores if not already done
        score_file_path = score_dir / f"{name}_scores.pkl"
        if not score_file_path.exists():
            scores = compute.calculate_match_score(
                preds,
                labels,
                frame_dir,
                real=False,
                name=name,
                dino_model=grounding_dino_model,
                debug_mode=debug
            )
            save_pickle(scores, score_file_path)

        # Append to main dataframe
        new_scores_df = compute.build_scoring_dataframe(str(score_file_path), pickle_loader=pickle)
        dataframe = pd.concat([dataframe, new_scores_df])

    return dataframe


# --- Workflow Functions ---

def process_synthetic_workflow(config: Dict[str, Any], save_root: Path, dino_model: Any) -> None:
    """Handles the execution logic for Synthetic data."""
    exp_date = config['experiment']['experiment_date']
    synth_source = config['experiment']['synth_source']
    overwrite = config['settings']['overwrite_evals']
    sim_threshold = config['settings']['similarity_threshold']
    debug = config['settings']['debug_mode']
    plot_metrics = config['settings']['plot_metrics']
    mappings = config['mappings']

    # Setup Paths
    label_root = ROOT / 'datasets' / 'synthetic_data'
    data_dir = ROOT / 'datasets' / 'synthetic_data' / synth_source

    if synth_source == 'gpt':
        label_dir = data_dir / 'images'
    else:
        label_dir = data_dir / 'labels'

    eval_dir = save_root / 'evaluations' / exp_date / synth_source
    eval_metrics_dir = eval_dir / 'eval_metrics'
    score_dir = eval_dir / 'scores'
    plot_dir = eval_dir / 'plots'

    for d in [eval_dir, eval_metrics_dir, score_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Run Evaluations
    data_root_path = Path(f"{save_root}/experiments/{exp_date.split('_')[0]}/synthetic")

    if synth_source == 'gpt':
        eval_synth_images(
            data_root_path / "gpt",
            label_root,
            eval_metrics_dir,
            overwrite=overwrite,
            sim_threshold=sim_threshold
        )
    elif synth_source == 'veo':
        eval_synth_video(
            data_root_path / "veo",
            label_root,
            eval_metrics_dir,
            overwrite=overwrite,
            sim_threshold=sim_threshold
        )
    else:
        raise ValueError(f"Media type {synth_source} not recognized.")

    # 2. Build Score Dataframe
    df = pd.DataFrame()
    if synth_source == 'gpt':
        for task in os.listdir(eval_metrics_dir):
            if '.' not in task:
                df = build_eval_dataframe_recursive(
                    df, eval_metrics_dir / task, data_dir, label_dir,
                    synth_source, score_dir, dino_model, debug, task=task
                )
    else:
        df = build_eval_dataframe_recursive(
            df, eval_metrics_dir, data_dir, label_dir,
            synth_source, score_dir, dino_model, debug
        )

    # 3. Save and Plot
    save_pickle(df, score_dir / "aggregated_scores.pkl")

    name_suffix = "Images" if synth_source == 'gpt' else "Video"
    compute.plot_performance_matrix(
        df, plot_metrics, plot_dir, mappings,
        name=f"Aggregated Synthetic {name_suffix}",
        fig_x=30
    )


def process_real_workflow(config: Dict[str, Any], save_root: Path, dino_model: Any) -> None:
    """Handles the execution logic for Real World data."""
    exp_date = config['experiment']['experiment_date']
    cap_date = config['experiment']['capture_date']
    overwrite = config['settings']['overwrite_evals']
    sim_threshold = config['settings']['similarity_threshold']
    debug = config['settings']['debug_mode']
    plot_metrics = config['settings']['plot_metrics']
    mappings = config['mappings']
    real_data_tasks = config['tasks']['real_world']

    data_dir = ROOT / 'datasets' / 'real_data'
    label_dir = data_dir / 'evaluations' / 'labels'
    results_root = Path(f'{save_root}/experiments/{exp_date}/real')

    eval_dir = save_root / 'evaluations' / exp_date / 'real'
    eval_metrics_dir = eval_dir / 'eval_metrics'
    score_dir = eval_dir / 'scores'
    plot_dir = eval_dir / 'plots'

    for d in [eval_dir, eval_metrics_dir, score_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    combined_data = pd.DataFrame()

    for group in real_data_tasks:
        scenario = group['name']
        for run in group['runs']:
            frame_dir = data_dir / 'experiments' / cap_date / scenario / run / 'cam0' / 'frames'

            if debug:
                debug_dir = data_dir / 'evaluations' / 'debugs' / scenario / run
                debug_dir.mkdir(parents=True, exist_ok=True)

            # 1. Run Evaluation
            real_world_evals(
                scenario, run, results_root, label_dir, eval_metrics_dir,
                overwrite=overwrite, sim_threshold=sim_threshold
            )

            # 2. Calculate Scores
            eval_pkl_path = eval_metrics_dir / f"{scenario}_{run}_eval.pkl"
            if not eval_pkl_path.exists():
                logging.warning(f"Evaluation file missing for {scenario}_{run}, skipping score calculation.")
                continue

            preds = load_pickle(eval_pkl_path)
            labels = load_pickle(label_dir / f"{scenario}_{run}_cam0_annotations_24fps.pkl")

            score_pkl_path = score_dir / f"{scenario}_{run}_scores.pkl"
            scores = compute.calculate_match_score(
                preds, labels, frame_dir,
                dino_model=dino_model, debug_mode=debug
            )
            save_pickle(scores, score_pkl_path)

            # 3. Plotting per run
            df = compute.build_scoring_dataframe(str(score_pkl_path), pickle_loader=pickle)
            run_plot_dir = plot_dir / f"{scenario}_{run}"
            run_plot_dir.mkdir(exist_ok=True)

            compute.plot_performance_matrix(
                df, plot_metrics, run_plot_dir, mappings, name=f'{scenario} {run}'
            )

            combined_data = pd.concat([combined_data, df], ignore_index=True)

    # 4. Save and Plot Aggregated
    save_pickle(combined_data, score_dir / "aggregated_scores.pkl")

    agg_plot_dir = plot_dir / "aggregated"
    agg_plot_dir.mkdir(exist_ok=True)

    compute.plot_performance_matrix(
        combined_data, plot_metrics, agg_plot_dir, mappings, name='Real Aggregated Scores'
    )


def process_all_workflow(config: Dict[str, Any], save_root: Path) -> None:
    """Handles the aggregation of all data types (Real + Synth Image + Synth Video)."""
    exp_date = config['experiment']['experiment_date']
    plot_metrics = config['settings']['plot_metrics']
    mappings = config['mappings']

    data_dir = save_root / 'evaluations' / exp_date
    plot_dir = data_dir / 'all'
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        real_data = load_pickle(data_dir / "real/scores/aggregated_scores.pkl")
        synth_img_data = load_pickle(data_dir / "gpt/scores/aggregated_scores.pkl")
        synth_video_data = load_pickle(data_dir / "veo/scores/aggregated_scores.pkl")
    except FileNotFoundError as e:
        logging.error(f"Error loading aggregated scores for 'all' workflow: {e}")
        return

    combined_data = pd.concat([real_data, synth_img_data, synth_video_data], ignore_index=True)
    save_pickle(combined_data, plot_dir / "aggregated_scores.pkl")

    compute.plot_performance_matrix(
        combined_data, plot_metrics, plot_dir, mappings,
        name='All_Aggregated_Scores', fig_x=80
    )


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    config = load_config(config_path=ROOT / "configs" / "evaluation_config.yaml")

    # Determine Save Root
    if config['paths']['save_root'] is not None:
        save_root = Path(config['paths']['save_root'])
    else:
        save_root = ROOT

    data_type = config['experiment']['data_type']
    debug = config['settings']['debug_mode']

    # Initialize GroundingDino Model if needed (used for debugging)
    grounding_dino = None if not debug else load_dino_model(config)

    # Dispatch based on data type
    logging.info(f"Starting execution for data type: {data_type}")

    if data_type == 'synthetic':
        process_synthetic_workflow(config, save_root, grounding_dino)
    elif data_type == 'real':
        process_real_workflow(config, save_root, grounding_dino)
    elif data_type == 'all':
        process_all_workflow(config, save_root)
    else:
        logging.error(f"Unknown data type: {data_type}")


if __name__ == '__main__':
    main()