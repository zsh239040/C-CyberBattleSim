#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
hyperopt_defender_marl.py
Hyper-parameter optimization for defender algorithms in MARL training.
Attacker stays fixed; defender hyperparams are tuned.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
import yaml
import optuna

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from cyberbattle.agents.multi_env_marl.train_agent_multi_env_marl import (  # noqa: E402
    setup_train_via_args_marl,
    train_multiagent,
)
from cyberbattle.utils.train_utils import check_args  # noqa: E402
from cyberbattle.utils.file_utils import save_yaml  # noqa: E402

script_dir = os.path.dirname(__file__)


def _make_hyperopt_logger(logs_folder: str, log_to_file: bool = True) -> logging.Logger:
    logger = logging.getLogger("hyperopt_defender")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_to_file:
        os.makedirs(logs_folder, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(logs_folder, "hyperopt.log"), mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _find_latest_hyperopt_folder(base_dir, prefix):
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for entry in os.listdir(base_dir):
        if entry.startswith(prefix):
            path = os.path.join(base_dir, entry)
            if os.path.isdir(path):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_hyperopt_logs_folder(args, base_dir):
    if args.resume_from:
        logs_folder = args.resume_from
        if not os.path.isabs(logs_folder):
            logs_folder = os.path.join(base_dir, "logs", logs_folder)
        if not os.path.isdir(logs_folder):
            raise ValueError(f"Resume folder does not exist: {logs_folder}")
        return logs_folder, True
    if args.resume_latest:
        logs_root = os.path.join(base_dir, "logs")
        prefix = f"hyperopt_{args.name}_" if args.name else "hyperopt_"
        logs_folder = _find_latest_hyperopt_folder(logs_root, prefix)
        if not logs_folder:
            raise ValueError(f"No hyperopt logs found with prefix {prefix} in {logs_root}")
        return logs_folder, True
    if args.name:
        logs_folder = os.path.join(
            base_dir, "logs", "hyperopt_" + args.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    else:
        logs_folder = os.path.join(base_dir, "logs", "hyperopt_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return logs_folder, False


map_name_to_sampler = {
    "grid": optuna.samplers.GridSampler,
    "random": optuna.samplers.RandomSampler,
    "tpe": optuna.samplers.TPESampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "GPS": optuna.samplers.GPSampler,
    "partial_fixed": optuna.samplers.PartialFixedSampler,
    "nsga2": optuna.samplers.NSGAIISampler,
    "nsga3": optuna.samplers.NSGAIIISampler,
    "qmc": optuna.samplers.QMCSampler,
    "bruteforce": optuna.samplers.BruteForceSampler,
}


def suggest_hyperparameters(trial, hyperparam_ranges):
    suggested_params = {}
    for param_name, param_config in hyperparam_ranges.items():
        if param_config["type"] == "categorical":
            suggested_params[param_name] = trial.suggest_categorical(param_name, param_config["values"])
        elif param_config["type"] == "float":
            suggested_params[param_name] = trial.suggest_float(
                param_name,
                param_config["low"],
                param_config["high"],
                log=param_config.get("log", False),
            )
        elif param_config["type"] == "int":
            suggested_params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])
    return suggested_params


def _extract_metric(run_metrics, metric_name):
    if not run_metrics:
        return None
    metrics = run_metrics[-1].get("defender")
    if not metrics:
        return None
    if metric_name == "defender/last10_avg_reward":
        return metrics.get("avg_reward")
    if metric_name == "defender/last_ep_reward":
        return metrics.get("last_reward")
    if metric_name == "defender/last10_avg_len":
        return metrics.get("avg_len")
    if metric_name == "defender/last_ep_len":
        return metrics.get("last_len")
    return metrics.get("avg_reward")


def _prepare_shared_envs(args, logs_folder: str, hyperopt_logger: logging.Logger):
    if not args.reuse_envs:
        return None
    logs_root = os.path.join(script_dir, "logs")
    shared_run = args.shared_envs_run or "shared_envs"
    shared_run_path = os.path.join(logs_folder, shared_run)
    shared_envs_path = os.path.join(shared_run_path, "envs")
    shared_rel = os.path.relpath(shared_run_path, logs_root)
    if os.path.isdir(shared_envs_path) and any(name.endswith(".pkl") for name in os.listdir(shared_envs_path)):
        hyperopt_logger.info("Reusing shared envs at %s", shared_envs_path)
        return shared_rel
    hyperopt_logger.info("Preparing shared envs at %s", shared_envs_path)
    args_copy = argparse.Namespace(**vars(args))
    args_copy.load_processed_envs = False
    args_copy.resume_from = None
    setup_train_via_args_marl(args_copy, logs_folder=shared_run_path)
    return shared_rel


def _run_single_trial(params_path: str, args, hyperopt_logger: logging.Logger, shared_envs_rel=None):
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Trial params not found: {params_path}")
    with open(params_path, "r") as f:
        suggested_params = yaml.safe_load(f) or {}
    hyperopt_logger.info("Replaying trial params from %s: %s", params_path, suggested_params)
    args_dict = vars(args).copy()
    replay_root = os.path.join(os.path.dirname(params_path), f"replay_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(replay_root, exist_ok=True)
    return objective(
        trial=None,
        hyperparam_ranges={},
        args_dict=args_dict,
        logs_folder=replay_root,
        hyperopt_logger=hyperopt_logger,
        suggested_params_override=suggested_params,
        trial_number_override=f"replay_{args.replay_trial or 'params'}",
        shared_envs_rel=shared_envs_rel,
    )


def objective(
    trial,
    hyperparam_ranges,
    args_dict,
    logs_folder,
    hyperopt_logger,
    suggested_params_override=None,
    trial_number_override=None,
    shared_envs_rel=None,
    seen_params=None,
):
    trial_args = argparse.Namespace(**args_dict)
    if suggested_params_override is None:
        suggested_params = suggest_hyperparameters(trial, hyperparam_ranges)
    else:
        suggested_params = suggested_params_override

    trial_number = trial_number_override
    if trial_number is None:
        trial_number = trial.number if trial is not None else "replay"

    hyperopt_logger.info("Trial %s suggested params: %s", trial_number, suggested_params)

    if seen_params is not None and suggested_params_override is None:
        params_key = tuple(sorted(suggested_params.items()))
        if params_key in seen_params:
            hyperopt_logger.warning("Trial %s duplicate params detected, pruning: %s", trial_number, suggested_params)
            raise optuna.TrialPruned()
        seen_params.add(params_key)

    trial_logs = os.path.join(logs_folder, f"trial_{trial_number}")
    os.makedirs(trial_logs, exist_ok=True)
    save_yaml(suggested_params, trial_logs, "defender_hyperparams.yaml")

    if shared_envs_rel:
        trial_args.load_processed_envs = shared_envs_rel
        trial_args.load_envs = None

    try:
        train_logger, logs_folder_run, envs_folder, config, train_ids, _val_ids = setup_train_via_args_marl(
            trial_args, logs_folder=trial_logs
        )
        # Also log to the trial's app.log for continuity with older runs
        train_logger.info("Trial %s suggested params: %s", trial_number, suggested_params)

        defender_cfg_path = trial_args.defender_config
        if not os.path.isabs(defender_cfg_path):
            defender_cfg_path = os.path.join(script_dir, defender_cfg_path)
        with open(defender_cfg_path, "r") as f:
            defender_cfg = yaml.safe_load(f) or {}

        defender_cfg["defender_algorithm"] = trial_args.defender_algorithm

        algo_cfg_path = trial_args.defender_algo_config
        if not os.path.isabs(algo_cfg_path):
            algo_cfg_path = os.path.join(script_dir, algo_cfg_path)
        with open(algo_cfg_path, "r") as f:
            algo_cfg = yaml.safe_load(f) or {}

        defender_cfg["defender_algorithm_hyperparams"] = algo_cfg.get(trial_args.defender_algorithm, {})
        if "policy_kwargs" in algo_cfg:
            defender_cfg["defender_policy_kwargs"] = algo_cfg["policy_kwargs"]

        for k, v in suggested_params.items():
            defender_cfg["defender_algorithm_hyperparams"][k] = v

        attacker_n_steps = config["algorithm_hyperparams"].get("n_steps")
        if attacker_n_steps is not None:
            defender_cfg["defender_algorithm_hyperparams"]["n_steps"] = attacker_n_steps

        run_metrics = train_multiagent(
            logs_folder_run,
            envs_folder,
            config,
            train_ids,
            defender_cfg,
            logger=train_logger,
            verbose=trial_args.verbose,
            return_metrics=True,
        )
    except Exception:
        error_path = os.path.join(trial_logs, "trial_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        hyperopt_logger.exception("Trial %s failed. Traceback saved to %s", trial_number, error_path)
        raise

    metric_value = _extract_metric(run_metrics, trial_args.metric_name)
    if metric_value is None:
        metric_value = float("-inf")
    # Mirror summary in trial app.log
    if 'train_logger' in locals():
        train_logger.info("Trial %s metric %s = %s", trial_number, trial_args.metric_name, metric_value)
    hyperopt_logger.info("Trial %s metric %s = %s", trial_number, trial_args.metric_name, metric_value)
    return metric_value


def hyperopt_defender(hyperparam_ranges, args, logs_folder, hyperopt_logger, shared_envs_rel=None):
    if args.optimization_type in map_name_to_sampler:
        sampler = map_name_to_sampler[args.optimization_type]()
    else:
        raise ValueError("Unsupported optimization type specified:", args.optimization_type)

    args_dict = vars(args).copy()
    study = optuna.create_study(
        direction=args.direction,
        sampler=sampler,
        storage=args.storage or f"sqlite:///{os.path.join(logs_folder, args.name + '.db')}",
        study_name=args.name,
        load_if_exists=True,
    )
    seen_params = set()
    if getattr(args, "avoid_duplicate_trials", True):
        for trial in study.trials:
            if trial.params:
                seen_params.add(tuple(sorted(trial.params.items())))
    study.optimize(
        lambda trial: objective(
            trial,
            hyperparam_ranges,
            args_dict,
            logs_folder,
            hyperopt_logger,
            shared_envs_rel=shared_envs_rel,
            seen_params=seen_params if getattr(args, "avoid_duplicate_trials", True) else None,
        ),
        n_trials=args.num_trials,
        catch=(Exception,),
    )
    hyperopt_logger.info("Best parameters found: %s", study.best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-optimize defender for MARL training")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "a2c", "rppo", "trpo", "ddpg", "sac", "td3", "tqc"],
        default="trpo",
        help="Attacker algorithm (used to build the MARL training config)",
    )
    parser.add_argument("--defender_algorithm", type=str, choices=["ppo", "a2c", "trpo"], default="ppo")
    parser.add_argument("--defender_config", type=str, default="config/defender_config.yaml")
    parser.add_argument("--defender_algo_config", type=str, default="config/defender_algo_config.yaml")

    parser.add_argument("--environment_type", type=str, choices=["continuous", "global", "local"], default="continuous")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument(
        "-nlp",
        "--nlp_extractor",
        type=str,
        choices=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT", "SecRoBERTa"],
        default="CySecBERT",
    )
    parser.add_argument("--holdout", action="store_true", default=True)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument(
        "-g",
        "--goal",
        type=str,
        default="control",
        choices=["control", "discovery", "disruption", "control_node", "discovery_node", "disruption_node"],
    )
    parser.add_argument("--name", default="DEFENDER", help="Name prefix for the hyperopt run")
    parser.add_argument("--load_envs", default="syntethic_deployment_20_graphs_100_nodes")
    parser.add_argument("--load_processed_envs", default=False)
    parser.add_argument("--static_seeds", action="store_true", default=True)
    parser.add_argument("--load_seeds", default="config")
    parser.add_argument("--random_seeds", action="store_true", default=False)
    parser.add_argument("--static_defender_agent", default=None, choices=["reimage", "events", None])
    parser.add_argument("-pca", "--pca_components", default=None, type=int)
    parser.add_argument("-v", "--verbose", default=1, type=int)
    parser.add_argument("--no_save_log_file", action="store_false", dest="save_log_file", default=True)
    parser.add_argument("--save_csv_file", action="store_true", default=False)
    parser.add_argument("--save_embeddings_csv_file", action="store_true", default=False)
    parser.add_argument("--train_config", type=str, default="config/train_config.yaml")
    parser.add_argument("--rewards_config", type=str, default="config/rewards_config.yaml")
    parser.add_argument("--algo_config", type=str, default="config/algo_config.yaml")

    parser.add_argument(
        "--optimization_type",
        type=str,
        choices=["grid", "random", "tpe", "cmaes", "GPS", "partial_fixed", "nsga2", "nsga3", "qmc", "bruteforce"],
        default="tpe",
    )
    parser.add_argument("--hyperparams_ranges_file", type=str, default="config/defender_hyperparams_ranges.yaml")
    parser.add_argument(
        "--metric_name",
        type=str,
        default="defender/last10_avg_reward",
        choices=[
            "defender/last10_avg_reward",
            "defender/last_ep_reward",
            "defender/last10_avg_len",
            "defender/last_ep_len",
        ],
    )
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--resume_latest", action="store_true", default=False)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:////tmp/defender.db)")
    parser.add_argument("--reuse_envs", action="store_true", default=True,
                        help="Reuse shared processed envs across trials to save disk")
    parser.add_argument("--no_reuse_envs", action="store_false", dest="reuse_envs")
    parser.add_argument("--shared_envs_run", type=str, default="shared_envs")
    parser.add_argument("--avoid_duplicate_trials", action="store_true", default=True,
                        help="Skip trials with parameter combinations already evaluated")
    parser.add_argument("--allow_duplicate_trials", action="store_false", dest="avoid_duplicate_trials")
    parser.add_argument("--replay_trial", type=int, default=None,
                        help="Replay a trial id using saved defender_hyperparams.yaml")
    parser.add_argument("--replay_trial_path", type=str, default=None,
                        help="Replay using a specific defender_hyperparams.yaml path")

    args = parser.parse_args()

    if args.resume_from and args.resume_latest:
        raise ValueError("Use only one of --resume_from or --resume_latest.")

    check_args(args)

    ranges_path = args.hyperparams_ranges_file
    if not os.path.isabs(ranges_path):
        ranges_path = os.path.join(script_dir, ranges_path)
    with open(ranges_path, "r") as file:
        hyperparams_ranges = yaml.safe_load(file) or {}

    hyperparams_ranges = hyperparams_ranges.get(args.defender_algorithm, {})

    logs_folder, resume_requested = _resolve_hyperopt_logs_folder(args, script_dir)
    hyperopt_logger = _make_hyperopt_logger(logs_folder, log_to_file=args.save_log_file)
    if resume_requested:
        hyperopt_logger.info("Resuming hyperopt run from %s", logs_folder)

    if not args.name:
        args.name = "hyperopt_defender_" + args.defender_algorithm

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    save_yaml(hyperparams_ranges, logs_folder, "defender_hyperparams_ranges.yaml")

    shared_envs_rel = _prepare_shared_envs(args, logs_folder, hyperopt_logger)

    if args.replay_trial_path or args.replay_trial is not None:
        if args.replay_trial_path:
            params_path = args.replay_trial_path
        else:
            params_path = os.path.join(logs_folder, f"trial_{args.replay_trial}", "defender_hyperparams.yaml")
        _run_single_trial(params_path, args, hyperopt_logger, shared_envs_rel=shared_envs_rel)
        sys.exit(0)

    hyperopt_defender(hyperparams_ranges, args, logs_folder, hyperopt_logger, shared_envs_rel=shared_envs_rel)
