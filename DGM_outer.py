import argparse
import datetime
import json
import math
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError

from prompts.self_improvement_prompt import find_selfimprove_eval_logs
from utils.common_utils import load_json_file
from utils.evo_utils import load_dgm_metadata, is_compiled_self_improve

SAFE_SWEBENCH_PRO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_swebench_pro_task_id(task_id):
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Invalid SWE-bench Pro task ID: {task_id!r}")
    if task_id in {".", ".."} or not SAFE_SWEBENCH_PRO_ID_RE.fullmatch(task_id):
        raise ValueError(f"Unsafe SWE-bench Pro task ID: {task_id!r}")
    return task_id


def initialize_run(
    output_dir,
    prevrun_dir=None,
    polyglot=False,
    polyglot_initial_dir=None,
    swebench_pro=False,
    swebench_pro_initial_dir=None,
):
    # Initialize archive
    start_gen_num = 0
    if not prevrun_dir:
        archive = ['initial']
    else:
        # Load previous run's archive
        metadata_path = os.path.join(prevrun_dir, "dgm_metadata.jsonl")
        metadata = load_dgm_metadata(metadata_path, last_only=True)
        archive = metadata['archive']
        start_gen_num = metadata['generation'] + 1

    # Copy cached initial version into experiment dir.
    # DGM expects the selected initial metadata to live at output_dir/initial.
    if swebench_pro:
        initial_folder_name = 'initial_swebench_pro'
        initial_source = swebench_pro_initial_dir or initial_folder_name
    elif polyglot:
        initial_folder_name = 'initial_polyglot'
        initial_source = polyglot_initial_dir or initial_folder_name
    else:
        initial_folder_name = 'initial'
        initial_source = initial_folder_name
    initial_dest = os.path.join(output_dir, "initial")
    if not prevrun_dir and not os.path.exists(initial_dest):
        if os.path.exists(initial_source):
            if not os.path.isdir(initial_source):
                raise RuntimeError(f"Initial source must be a directory: {initial_source}")
            shutil.copytree(initial_source, initial_dest)
        else:
            raise RuntimeError(f"Error: Need to properly configure evaluation results for the initial version: {initial_source}")

    return archive, start_gen_num


def load_task_ids_file(path):
    data = load_json_file(path)
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return [validate_swebench_pro_task_id(item) for item in data]
    if isinstance(data, dict):
        if isinstance(data.get("task_ids"), list):
            return [validate_swebench_pro_task_id(item) for item in data["task_ids"] if isinstance(item, str)]
        if isinstance(data.get("tasks"), list):
            task_ids = []
            for item in data["tasks"]:
                if not isinstance(item, dict):
                    continue
                task_id = item.get("task_id") or item.get("instance_id")
                if isinstance(task_id, str):
                    task_ids.append(validate_swebench_pro_task_id(task_id))
            return task_ids
    raise ValueError(f"Unsupported task ID file: {path}")

def any_exceeding_context_length(output_dir, commit_id, instance_ids):
    """
    Check if any of the issues have exceeded the context length.
    """
    for instance_id in instance_ids:
        md_logs, _, _, _ = find_selfimprove_eval_logs(instance_id, output_dir, commit_id, filter=False)
        md_log = md_logs[0]
        error_str = "Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}"
        # Repeated error_str means no attempt to fix it
        if f'{error_str}\n{error_str}' in md_log:
            return True
    return False

def choose_selfimproves(output_dir, archive, selfimprove_size, method='random', run_baseline=None, polyglot=False):
    """
    Choose self-improve attempts for the current generation.
    """
    selfimprove_entries = []

    # Get parent candidates
    candidates = {}
    for commit in archive:
        try:
            metadata_path = os.path.join(output_dir, commit, "metadata.json")
            metadata = load_json_file(metadata_path)
            candidates[commit] = {
                'accuracy_score': metadata['overall_performance']['accuracy_score'],
                'total_unresolved_ids': metadata['overall_performance']['total_unresolved_ids'],
                'total_emptypatch_ids': metadata['overall_performance']['total_emptypatch_ids'],
                'total_resolved_ids': metadata['overall_performance']['total_resolved_ids'],
                'children_count': 0,
            }
            # update children count, parent should already be in the archive
            if commit != 'initial':
                parent_commit = metadata['parent_commit']
                candidates[parent_commit]['children_count'] += 1
        except Exception as e:
            # probably because swe-eval failed, generated code did not compile, etc.
            print(f"{commit} not eligible for being a parent: {e}")
            continue

    # Choose parents based on method and baseline
    if run_baseline == 'no_darwin':
        # Always take the last commit
        commits = list(candidates.keys())
        parent_commits = commits[-1:]
    elif method == 'score_prop':
        # Choose parents based on score
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        probabilities = [score / sum(scores) for score in scores]
        print(commits)
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'score_child_prop':
        # Choose parents based on score and the number of children
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        children_counts = [candidates[commit]['children_count'] for commit in commits]
        children_counts = [1 / (1 + count) for count in children_counts]
        probabilities = [score * count for score, count in zip(scores, children_counts)]
        probabilities = [prob / sum(probabilities) for prob in probabilities]
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'best':
        # Choose parents with the best score
        sorted_commits = sorted(candidates, key=lambda x: candidates[x]['accuracy_score'])
        parent_commits = sorted_commits[:min(selfimprove_size, len(sorted_commits))]
        if len(parent_commits) < selfimprove_size:
            parent_commits.extend(random.choices(parent_commits, k=selfimprove_size - len(parent_commits)))
    else:
        # Choose parents randomly
        parent_commits = random.choices(list(candidates.keys()), k=selfimprove_size)

    # Choose entries for each parent
    for parent_commit in parent_commits:
        empty_ids = candidates[parent_commit]['total_emptypatch_ids']
        resolved_ids = candidates[parent_commit]['total_resolved_ids']
        unresolved_ids = candidates[parent_commit]['total_unresolved_ids']

        if polyglot:
            entry_ids = empty_ids + unresolved_ids
            if not entry_ids:
                entry_ids = resolved_ids + empty_ids + unresolved_ids
        else:
            num_total_ids = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)

            # Solve empty patches
            if len(empty_ids) >= 0.1 * num_total_ids and random.random() < 0.25:
                entry = 'solve_empty_patches'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Solve stochasticity
            if random.random() < 0.25:
                entry = 'solve_stochasticity'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Solve context length
            if any_exceeding_context_length(output_dir, parent_commit, empty_ids + unresolved_ids) and \
                random.random() < 0.25:
                entry = 'solve_contextlength'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Choose a random unresolved entry
            if unresolved_ids == 0:
                continue
            entry_ids = unresolved_ids
        entry = random.choice(entry_ids)
        selfimprove_entries.append((parent_commit, entry))

    return selfimprove_entries

def filter_compiled(run_ids, output_dir, num_swe_issues=[], logger=None):
    """
    Filter out runs that did not compile or have all empty patches.
    """
    run_ids_compiled = []

    logger.info(f"num_swe_issues: {num_swe_issues}")
    for run_id in run_ids:
        metadata_path = os.path.join(output_dir, run_id, "metadata.json")
        metadata = load_json_file(metadata_path)
        logger.info(f"{run_id} metadata: {metadata}")
        if is_compiled_self_improve(metadata, num_swe_issues=num_swe_issues, logger=logger):
            run_ids_compiled.append(run_id)
    return run_ids_compiled

def get_original_score(output_dir):
    """
    Get the original score from the initial version.
    """
    metadata = load_json_file(os.path.join(output_dir, "initial", "metadata.json"))
    return metadata["overall_performance"]["accuracy_score"]

def update_archive(output_dir, archive, new_ids, method='keep_all', noise_leeway=0.1):
    """
    Update the archive with the new self-improve runs.
    """
    if method == 'keep_better':
        # keep only better ones
        original_score = get_original_score(output_dir) - noise_leeway
        for run_id in new_ids:
            metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
            score = metadata["overall_performance"]["accuracy_score"]
            if score >= original_score:
                archive.append(run_id)
    else:
        # keep everything
        archive += new_ids

    return archive

def get_full_eval_threshold(output_dir, archive, full_eval_count=None):
    """
    Get the threshold for full evaluation.
    """
    archive_scores = []
    num_full_eval = full_eval_count or sum(len(load_json_file(f"./swe_bench/subsets/{size}.json"))
                                           for size in ['small', 'medium', 'big'])

    # Get original score
    original_score = get_original_score(output_dir)
    archive_scores.append(original_score)

    # Get scores from the archive
    for run_id in archive:
        metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
        total_submitted_instances = metadata["overall_performance"]["total_submitted_instances"]
        # Skip if node did not have full evaluation
        if total_submitted_instances < num_full_eval * 0.9:
            continue
        score = metadata["overall_performance"]["accuracy_score"]
        archive_scores.append(score)

    # Get threshold, second highest score
    threshold = sorted(archive_scores, reverse=True)[1] if len(archive_scores) > 1 else archive_scores[0]
    # Ensure threshold is at least 0.4
    threshold = max(threshold, 0.4)

    return threshold

def main():
    parser = argparse.ArgumentParser(description="Darwin Godel Machine!")
    parser.add_argument("--max_generation", type=int, default=80, help="Maximum number of evolution iterations.")
    parser.add_argument("--selfimprove_size", type=int, default=2, help="Number of self-improvements attempts per DGM generation.")
    parser.add_argument("--selfimprove_workers", type=int, default=2, help="Number of parallel workers for self-improvement attempts.")
    parser.add_argument(
        "--choose_selfimproves_method", type=str, default='score_child_prop',
        choices=['random', 'score_prop', 'score_child_prop', 'best'],
        help="Method to choose self-improve attempts.",
    )
    parser.add_argument("--continue_from", type=str, default=None, help="Directory to continue the run from.")
    parser.add_argument("--update_archive", type=str, default='keep_all', choices=['keep_better', 'keep_all'], help="Method to update the archive.")
    # self-improve arguments
    parser.add_argument("--num_swe_evals", type=int, default=1, help="Number of repeated SWE evaluations to run for each self-improve attempt.")
    parser.add_argument('--post_improve_diagnose', default=False, action='store_true', help='Diagnose the self-improvement after evaluation')
    parser.add_argument("--shallow_eval", default=False, action='store_true', help="Run single shallow evaluation for self-improvement on swe.")
    parser.add_argument("--polyglot", default=False, action='store_true', help="Run single shallow evaluation for self-improvement on swe.")
    parser.add_argument("--swebench_pro", default=False, action='store_true', help="Run DGM self-improvement on SWE-bench Pro.")
    parser.add_argument("--eval_noise", type=float, default=0.1, help="Noise leeway for evaluation.")
    parser.add_argument("--no_full_eval", default=False, action='store_true', help="Do not run full evaluation on swe if a node is the top N highest performing.")
    parser.add_argument(
        "--polyglot_small_subset",
        type=str,
        default=os.getenv(
            "DGM_POLYGLOT_SMALL_SUBSET",
            "../../benchmarks/polyglot/task_maps/polyglot_medium_50_seed0_ids.json",
        ),
        help="Polyglot first-stage subset JSON. Override to reuse prepared benchmark task maps.",
    )
    parser.add_argument(
        "--polyglot_medium_subset",
        type=str,
        default=os.getenv(
            "DGM_POLYGLOT_MEDIUM_SUBSET",
            "../../benchmarks/polyglot/task_maps/polyglot_medium_50_seed0_ids.json",
        ),
        help="Polyglot second-stage subset JSON. Override to reuse prepared benchmark task maps.",
    )
    parser.add_argument(
        "--polyglot_initial_dir",
        type=str,
        default=os.getenv("DGM_POLYGLOT_INITIAL_DIR"),
        help="Initial Polyglot metadata directory to seed output_dgm/<run>/initial.",
    )
    parser.add_argument(
        "--swebench_pro_dataset_path",
        type=str,
        default=os.getenv(
            "DGM_SWEBENCH_PRO_DATASET",
            "../../benchmarks/swebench_pro/dataset/test.jsonl",
        ),
        help="SWE-bench Pro raw sample JSONL.",
    )
    parser.add_argument(
        "--swebench_pro_task_map",
        type=str,
        default=os.getenv(
            "DGM_SWEBENCH_PRO_TASK_MAP",
            "../../benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed0_v1.json",
        ),
        help="SWE-bench Pro task map JSON.",
    )
    parser.add_argument(
        "--swebench_pro_initial_dir",
        type=str,
        default=os.getenv("DGM_SWEBENCH_PRO_INITIAL_DIR"),
        help="Initial SWE-bench Pro metadata directory to seed output_dgm/<run>/initial.",
    )
    parser.add_argument(
        "--swebench_pro_eval_source",
        type=str,
        default=os.getenv("DGM_SWEBENCH_PRO_EVAL_SOURCE", "../../third_party/SWE-bench_Pro-os"),
        help="Path to the official SWE-bench Pro evaluator checkout.",
    )
    parser.add_argument(
        "--swebench_pro_scripts_dir",
        type=str,
        default=os.getenv("DGM_SWEBENCH_PRO_SCRIPTS_DIR"),
        help="Path to SWE-bench Pro run_scripts. Defaults to <eval_source>/run_scripts.",
    )
    parser.add_argument(
        "--swebench_pro_dockerhub_username",
        type=str,
        default=os.getenv("DGM_SWEBENCH_PRO_DOCKERHUB_USERNAME", "jefzda"),
        help="Docker Hub username containing sweap-images.",
    )
    parser.add_argument(
        "--swebench_pro_docker_platform",
        type=str,
        default=os.getenv("DGM_SWEBENCH_PRO_DOCKER_PLATFORM"),
        help="Optional Docker platform override, e.g. linux/amd64.",
    )
    parser.add_argument(
        "--swebench_pro_block_network",
        default=False,
        action='store_true',
        help="Block network inside official SWE-bench Pro evaluation containers.",
    )
    parser.add_argument(
        "--swebench_pro_no_local_docker",
        default=False,
        action='store_true',
        help="Use Modal instead of local Docker for official SWE-bench Pro evaluation.",
    )
    parser.add_argument(
        "--swebench_pro_stage_size",
        type=int,
        default=int(os.getenv("DGM_SWEBENCH_PRO_STAGE_SIZE", "10")),
        help="First-stage task count when not using --shallow_eval.",
    )
    # baselines
    parser.add_argument("--run_baseline", type=str, default=None, choices=['no_selfimprove', 'no_darwin'], help="Baseline to run.")
    args = parser.parse_args()

    if args.polyglot and args.swebench_pro:
        parser.error("--polyglot and --swebench_pro cannot both be set")

    from self_improve_step import self_improve
    from utils.docker_utils import setup_logger

    # Variables for this DGM run
    if not args.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    else:
        run_id = os.path.basename(args.continue_from)

    output_dir = os.path.join("./output_dgm", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize
    archive, start_gen_num = initialize_run(
        output_dir,
        prevrun_dir=args.continue_from,
        polyglot=args.polyglot,
        polyglot_initial_dir=args.polyglot_initial_dir,
        swebench_pro=args.swebench_pro,
        swebench_pro_initial_dir=args.swebench_pro_initial_dir,
    )

    # SWE issues to consider
    swebench_pro_all_ids = None
    if args.swebench_pro:
        swebench_pro_all_ids = load_task_ids_file(args.swebench_pro_task_map)
        if args.shallow_eval:
            swe_issues_sm = swebench_pro_all_ids
            swe_issues_med = []
        else:
            stage_size = max(1, min(args.swebench_pro_stage_size, len(swebench_pro_all_ids)))
            swe_issues_sm = swebench_pro_all_ids[:stage_size]
            swe_issues_med = swebench_pro_all_ids[stage_size:]
    elif not args.polyglot:
        swe_issues_sm = load_json_file("./swe_bench/subsets/small.json")
        swe_issues_med = load_json_file("./swe_bench/subsets/medium.json")
    else:
        swe_issues_sm = load_json_file(args.polyglot_small_subset)
        swe_issues_med = load_json_file(args.polyglot_medium_subset)

    # Set up logger
    logger = setup_logger(os.path.join(output_dir, "dgm_outer.log"))
    logger.info(f"Starting DGM run {run_id} with arguments: {vars(args)}")
    logger.info(f"Archive: {archive}")
    test_more_threshold = 0.4
    # Run the DGM
    for gen_num in range(start_gen_num, args.max_generation):
        # Choose self-improve attempts
        selfimprove_entries = choose_selfimproves(
            output_dir, archive, args.selfimprove_size,
            method=args.choose_selfimproves_method,
            run_baseline=args.run_baseline,
            polyglot=args.polyglot,
        )
        logger.info(f"Self-improve entries for generation {gen_num}: {selfimprove_entries}")

        # Run self-improvement processes
        selfimprove_ids = []
        with ThreadPoolExecutor(max_workers=args.selfimprove_workers) as executor:
            futures = [
                executor.submit(
                    self_improve,
                    parent_commit=parent_commit,
                    output_dir=output_dir,
                    force_rebuild=False,
                    num_evals=args.num_swe_evals,
                    post_improve_diagnose=args.post_improve_diagnose,
                    entry=entry,
                    test_task_list=swe_issues_sm,
                    test_more_threshold=None if args.shallow_eval else test_more_threshold,
                    test_task_list_more=None if args.shallow_eval else swe_issues_med,
                    polyglot=args.polyglot,
                    swebench_pro=args.swebench_pro,
                    swebench_pro_dataset_path=args.swebench_pro_dataset_path,
                    swebench_pro_task_map=args.swebench_pro_task_map,
                    swebench_pro_eval_source=args.swebench_pro_eval_source,
                    swebench_pro_scripts_dir=args.swebench_pro_scripts_dir,
                    swebench_pro_dockerhub_username=args.swebench_pro_dockerhub_username,
                    swebench_pro_use_local_docker=not args.swebench_pro_no_local_docker,
                    swebench_pro_docker_platform=args.swebench_pro_docker_platform,
                    swebench_pro_block_network=args.swebench_pro_block_network,
                    full_eval_threshold=None if args.no_full_eval else get_full_eval_threshold(
                        output_dir,
                        archive,
                        full_eval_count=len(swebench_pro_all_ids) if swebench_pro_all_ids else None,
                    ),
                    run_baseline=args.run_baseline,
                )
                for parent_commit, entry in selfimprove_entries
            ]

            for future in as_completed(futures):
                try:
                    # Added timeout to avoid hanging indefinitely (1.5 h here)
                    metadata = future.result(timeout=1.5*60*60)
                    selfimprove_ids.append(metadata['run_id'])
                except TimeoutError:
                    logger.error("Self-improvement attempt timed out.")
                    future.cancel()  # Optionally cancel the future if it's still running
                except Exception as e:
                    import traceback
                    logger.error(f"Self-improvement step failed: {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Update archive
        logger.info(f"Updating archive for generation {gen_num}")
        selfimprove_ids_compiled = filter_compiled(
            selfimprove_ids,
            output_dir,
            num_swe_issues=[len(swe_issues_sm)] if args.shallow_eval else [len(swe_issues_sm), len(swe_issues_med)], logger=logger
        )
        archive = update_archive(output_dir, archive, selfimprove_ids_compiled, method=args.update_archive, noise_leeway=args.eval_noise)

        # Save DGM state
        with open(os.path.join(output_dir, "dgm_metadata.jsonl"), "a") as f:
            f.write(json.dumps({
                "generation": gen_num,
                "selfimprove_entries": selfimprove_entries,
                "children": selfimprove_ids,
                "children_compiled": selfimprove_ids_compiled,
                "archive": archive,
            }, indent=2) + "\n")

if __name__ == "__main__":
    main()
