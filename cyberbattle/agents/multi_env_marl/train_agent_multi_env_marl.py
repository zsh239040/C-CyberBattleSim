# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    train_agent_multi_env_marl.py
    Multi-environment multi-agent training script for C-CyberBattleSim.
    Attacker + Defender trained jointly using a shared environment per worker.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import re
import shutil
import sys
import zipfile
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.logger import Logger, HumanOutputFormat, CSVOutputFormat, TensorBoardOutputFormat
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
cyberbattle_root = os.path.join(project_root, "cyberbattle")
sys.path.insert(0, project_root)

from cyberbattle.utils.train_utils import replace_with_classes, check_args, clean_config_save, algorithm_models, reccurrent_algorithms  # noqa: E402
from cyberbattle.utils.math_utils import linear_schedule, set_seeds  # noqa: E402
from cyberbattle.utils.file_utils import extract_metric_data, load_yaml, save_yaml  # noqa: E402
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv  # noqa: E402
from cyberbattle.utils.log_utils import setup_logging  # noqa: E402
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs, wrap_graphs_to_global_envs, wrap_graphs_to_local_envs  # noqa: E402
from cyberbattle._env.static_defender import ScanAndReimageCompromisedMachines, ExternalRandomEvents  # noqa: E402
from cyberbattle.gae.model import GAEEncoder  # noqa: E402

from cyberbattle.agents.multi_env_marl.env_wrappers.environment_event_source import EnvironmentEventSource  # noqa: E402
from cyberbattle.agents.multi_env_marl.env_wrappers.shared_env import SharedResetEnv  # noqa: E402
from cyberbattle.agents.multi_env_marl.env_wrappers.attacker_wrapper import AttackerEnvWrapper  # noqa: E402
from cyberbattle.agents.multi_env_marl.env_wrappers.defender_wrapper import DefenderEnvWrapper  # noqa: E402
from cyberbattle.agents.multi_env_marl.multiagent.baseline_agent import BaselineMultiAgent  # noqa: E402
from cyberbattle.agents.multi_env_marl.multiagent.simple_agent import SimpleMultiAgent  # noqa: E402
from cyberbattle.agents.multi_env_marl.multiagent.marl_algorithm import learn_multi  # noqa: E402
from cyberbattle.agents.multi_env_marl.policies import FactorizedDefenderPolicy  # noqa: E402


torch.set_default_dtype(torch.float32)
script_dir = os.path.dirname(__file__)


ON_POLICY_ALGOS = {"ppo", "a2c", "trpo"}


def _as_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _get_tensorboard_dir(logs_folder, algorithm, run_id):
    if algorithm != "rppo":
        return os.path.join(logs_folder, f"{algorithm.upper()}_{run_id}")
    return os.path.join(logs_folder, f"RecurrentPPO_{run_id}")


def _resolve_resume_logs_folder(resume_from, algorithm, goal, nlp_extractor):
    logs_folder = resume_from
    if not os.path.isabs(logs_folder):
        logs_folder = os.path.join(script_dir, "logs", logs_folder)
    if not os.path.isdir(logs_folder):
        raise ValueError(f"Resume folder does not exist: {logs_folder}")
    expected_run_folder = f"{algorithm.upper()}_x_{goal}_{nlp_extractor}"
    if os.path.basename(logs_folder) != expected_run_folder:
        candidate = os.path.join(logs_folder, expected_run_folder)
        if os.path.isdir(candidate):
            logs_folder = candidate
        else:
            raise ValueError(
                f"Resume folder does not match expected run folder {expected_run_folder}: {logs_folder}"
            )
    return logs_folder


def _run_completion_flag_path(logs_folder, run_id):
    return os.path.join(logs_folder, f"run_{run_id}_complete.txt")


def _find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    pattern = re.compile(r"checkpoint_(\d+)_steps\.zip$")
    latest_step = -1
    latest_path = None
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if not match:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_path = os.path.join(checkpoint_dir, filename)
    return latest_path


def _resolve_model_path(model_path: str) -> str:
    candidate = str(model_path).strip()
    if not candidate:
        raise ValueError("Empty model path was provided.")

    candidate = os.path.expanduser(candidate)
    paths = []
    if os.path.isabs(candidate):
        paths.append(candidate)
    else:
        paths.append(candidate)
        paths.append(os.path.join(script_dir, candidate))
        paths.append(os.path.join(script_dir, "logs", candidate))

    checked = []
    for path in paths:
        abspath = os.path.abspath(path)
        if abspath in checked:
            continue
        checked.append(abspath)
        if os.path.isfile(abspath):
            return abspath

    tried = "\n".join(checked)
    raise ValueError(f"Model path not found: {model_path}\nTried:\n{tried}")


def _load_model_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    if not str(model_path).lower().endswith(".zip"):
        return None
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            if "data" not in archive.namelist():
                return None
            return json.loads(archive.read("data").decode("utf-8"))
    except Exception:
        return None


def _detect_algorithm_from_path(model_path: str) -> Optional[str]:
    normalized = model_path.replace("\\", "/")
    match = re.search(r"(?i)(?:^|[/_\-])(rppo|trpo|ppo|a2c)(?:[/_\-]|$)", normalized)
    if match:
        return match.group(1).lower()
    return None


def _detect_on_policy_algorithm_from_model(model_path: str) -> Optional[str]:
    metadata = _load_model_metadata(model_path)
    if metadata:
        keys = set(metadata.keys())
        policy_kwargs = metadata.get("policy_kwargs")
        if isinstance(policy_kwargs, dict) and (
            "lstm_hidden_size" in policy_kwargs or "n_lstm_layers" in policy_kwargs
        ):
            return "rppo"
        if {"cg_max_steps", "line_search_max_iter", "n_critic_updates", "sub_sampling_factor"} & keys:
            return "trpo"
        if {"n_epochs", "clip_range", "clip_range_vf"} & keys:
            return "ppo"
        if {"rms_prop_eps", "use_rms_prop"} & keys:
            return "a2c"
    return _detect_algorithm_from_path(model_path)


def _split_env_ids(train_ids, num_parallel_envs, sampling_mode, seed, shuffle):
    if sampling_mode == "random":
        return [list(train_ids) for _ in range(num_parallel_envs)]
    rng = random.Random(seed)
    env_ids = list(train_ids)
    if shuffle:
        rng.shuffle(env_ids)
    groups = [[] for _ in range(num_parallel_envs)]
    for idx, env_id in enumerate(env_ids):
        groups[idx % num_parallel_envs].append(env_id)
    return groups


def _make_shared_env(rank, env_ids, envs_folder, csv_folder, config, seed, switch_interval):
    def _init():
        set_seeds(seed + rank)
        env = RandomSwitchEnv(
            env_ids,
            switch_interval,
            envs_folder=envs_folder,
            csv_folder=None,
            save_to_csv=False,
            save_to_csv_interval=config["save_csv_file"],
            save_embeddings=config["save_embeddings_csv_file"],
            verbose=config["verbose"],
            training_non_terminal_mode=_as_bool(config.get("marl_non_terminal_training", False)),
            training_disable_terminal_rewards=_as_bool(
                config.get("marl_disable_terminal_rewards", config.get("marl_non_terminal_training", False))
            ),
        )
        if config["save_csv_file"]:
            env_csv_folder = os.path.join(csv_folder, f"env_{rank}")
            env.update_csv_folder(
                env_csv_folder,
                filename=f"logs_env_{rank}.csv",
                save_embeddings=config["save_embeddings_csv_file"],
                save_to_csv_interval=config["save_csv_file"],
            )
        return SharedResetEnv(
            env,
            proportional_cutoff_coefficient=config.get("proportional_cutoff_coefficient"),
        )

    return _init


def _scan_defender_bounds(
    envs_folder: str,
    env_ids,
    firewall_rule_limit: Optional[int] = None,
    firewall_rule_source: str = "vulnerabilities",
    service_action_limit: Optional[int] = None,
    firewall_rule_min_support: Optional[int] = None,
) -> Tuple[int, int, List[str], float]:
    max_nodes = 1
    max_total_services = 1
    port_counts: Dict[str, int] = {}
    total_port_events = 0
    covered_port_events = 0
    firewall_rule_source = str(firewall_rule_source or "vulnerabilities").lower()
    service_action_limit = int(service_action_limit) if service_action_limit is not None else None
    firewall_rule_min_support = int(firewall_rule_min_support) if firewall_rule_min_support is not None else None
    for env_id in env_ids:
        path = os.path.join(envs_folder, f"{env_id}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            env = pickle.load(f)
        base_env = env
        nodes = list(base_env.environment.nodes)
        max_nodes = max(max_nodes, len(nodes))
        total_services = 0
        for node_id in nodes:
            node_info = base_env.get_node(node_id)
            total_services += len(node_info.services)
            if firewall_rule_source in {"services", "both"}:
                for service_idx, service in enumerate(node_info.services):
                    if service_action_limit is not None and service_idx >= service_action_limit:
                        break
                    port_name = service.name
                    port_counts[port_name] = port_counts.get(port_name, 0) + 1
                    total_port_events += 1
            if firewall_rule_source in {"vulnerabilities", "both"}:
                for vuln in node_info.vulnerabilities.values():
                    port_name = vuln.port
                    port_counts[port_name] = port_counts.get(port_name, 0) + 1
                    total_port_events += 1
        max_total_services = max(max_total_services, total_services)
        del env
    sorted_ports = sorted(port_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if firewall_rule_min_support is not None and firewall_rule_min_support > 1:
        sorted_ports = [item for item in sorted_ports if item[1] >= firewall_rule_min_support]
    if firewall_rule_limit is not None and firewall_rule_limit > 0:
        sorted_ports = sorted_ports[: firewall_rule_limit]
    firewall_ports = [port for port, _ in sorted_ports]
    if total_port_events:
        covered_port_events = sum(port_counts[p] for p in firewall_ports if p in port_counts)
    coverage = float(covered_port_events) / float(total_port_events) if total_port_events else 0.0
    return max_nodes, max_total_services, firewall_ports, coverage


def _build_vec_envs(logs_folder, envs_folder, config, train_ids, seed, defender_cfg, logger=None):
    if not train_ids:
        raise ValueError("No training environments available.")

    requested_parallel = config["num_parallel_envs"]
    num_parallel_envs = min(requested_parallel, len(train_ids))
    if requested_parallel > len(train_ids) and logger:
        logger.info(
            "Requested %d parallel envs, but only %d training envs are available. Capping to %d.",
            requested_parallel,
            len(train_ids),
            num_parallel_envs,
        )
    config["num_parallel_envs"] = num_parallel_envs

    env_id_groups = _split_env_ids(
        train_ids,
        num_parallel_envs,
        config["env_sampling_mode"],
        seed,
        config.get("shuffle_train_envs", True),
    )

    firewall_rule_limit = defender_cfg.get("defender_firewall_rule_limit")
    firewall_rule_source = defender_cfg.get("defender_firewall_rule_source", "vulnerabilities")
    service_action_limit = defender_cfg.get("defender_service_action_limit", 3)
    firewall_rule_min_support = defender_cfg.get("defender_firewall_rule_min_support")
    max_nodes, max_total_services, firewall_ports, firewall_coverage = _scan_defender_bounds(
        envs_folder,
        train_ids,
        firewall_rule_limit=firewall_rule_limit,
        firewall_rule_source=firewall_rule_source,
        service_action_limit=service_action_limit,
        firewall_rule_min_support=firewall_rule_min_support,
    )
    if not firewall_ports:
        firewall_ports = list(DefenderEnvWrapper.firewall_rule_list)
        if logger:
            logger.warning(
                "Defender firewall scan returned empty list; fallback to default ports %s.",
                firewall_ports,
            )
    defender_cfg["max_nodes"] = defender_cfg.get("max_nodes") or max_nodes
    defender_cfg["max_total_services"] = defender_cfg.get("max_total_services") or max_total_services
    defender_cfg["defender_firewall_rule_list"] = firewall_ports
    if logger:
        logger.info(
            "Defender firewall ports=%d coverage=%.3f (limit=%s)",
            len(firewall_ports),
            firewall_coverage,
            str(firewall_rule_limit),
        )

    csv_folder = os.path.join(logs_folder, "csv")
    shared_envs = [
        _make_shared_env(
            rank,
            env_id_groups[rank],
            envs_folder,
            csv_folder,
            config,
            seed,
            config["switch_interval"],
        )()
        for rank in range(num_parallel_envs)
    ]

    attacker_wrappers = []
    defender_wrappers = []
    non_terminal_training = _as_bool(config.get("marl_non_terminal_training", False))
    non_terminal_max_timesteps = config.get("marl_non_terminal_max_timesteps")
    if non_terminal_max_timesteps in {0, "0", "", False}:
        non_terminal_max_timesteps = None
    elif non_terminal_max_timesteps is not None:
        non_terminal_max_timesteps = int(non_terminal_max_timesteps)
    attacker_max_timesteps = config.get("episode_iterations")
    defender_max_timesteps = defender_cfg.get("defender_max_timesteps", config.get("episode_iterations", 200))
    if non_terminal_training:
        attacker_max_timesteps = non_terminal_max_timesteps
        defender_max_timesteps = non_terminal_max_timesteps
    deadlock_patience = int(config.get("marl_deadlock_patience", 0) or 0) if non_terminal_training else 0
    deadlock_on_no_owned_running = (
        _as_bool(config.get("marl_deadlock_reset_on_no_owned_running", False)) if non_terminal_training else False
    )
    for idx, shared_env in enumerate(shared_envs):
        event_source = EnvironmentEventSource()
        attacker_log_summary = config.get("attacker_log_episode_summary", True)
        attacker_log_summary = _as_bool(attacker_log_summary, True)
        attacker_wrappers.append(
            AttackerEnvWrapper(
                shared_env,
                event_source=event_source,
                max_timesteps=attacker_max_timesteps,
                training_non_terminal_mode=non_terminal_training,
                deadlock_patience=deadlock_patience,
                deadlock_reset_on_no_owned_running=deadlock_on_no_owned_running,
                log_episode_end=False,
                log_episode_summary=bool(attacker_log_summary),
                episode_log_prefix=f"[env={idx}]",
            )
        )
        defender_log_summary = defender_cfg.get("defender_log_episode_summary", True)
        defender_log_summary = _as_bool(defender_log_summary, True)
        defender_wrappers.append(
            DefenderEnvWrapper(
                shared_env,
                attacker_reward_store=attacker_wrappers[-1],
                event_source=event_source,
                max_nodes=defender_cfg["max_nodes"],
                max_total_services=defender_cfg["max_total_services"],
                service_action_limit=defender_cfg.get("defender_service_action_limit", 3),
                firewall_rule_list=defender_cfg.get("defender_firewall_rule_list"),
                max_timesteps=defender_max_timesteps,
                invalid_action_reward=defender_cfg.get("defender_invalid_action_reward", 0),
                invalid_action_autocorrect=defender_cfg.get("defender_invalid_action_autocorrect", True),
                invalid_action_autocorrect_penalty_scale=defender_cfg.get(
                    "defender_invalid_action_autocorrect_penalty_scale", 0.2
                ),
                reset_on_constraint_broken=defender_cfg.get("defender_reset_on_constraint_broken", False),
                loss_reward=defender_cfg.get("defender_loss_reward", -5000.0),
                defender_maintain_sla=defender_cfg.get("defender_maintain_sla", 0.6),
                sla_worsening_penalty_scale=defender_cfg.get("defender_sla_worsening_penalty_scale", 200.0),
                availability_delta_scale=defender_cfg.get("defender_availability_delta_scale", 0.0),
                owned_ratio_delta_scale=defender_cfg.get("defender_owned_ratio_delta_scale", 0.0),
                active_intrusion_step_penalty=defender_cfg.get("defender_active_intrusion_step_penalty", 10.0),
                attacker_reward_mode=defender_cfg.get("defender_attacker_reward_mode", "all"),
                attacker_reward_scale=defender_cfg.get("defender_attacker_reward_scale", 1.0),
                candidate_k_hop=defender_cfg.get("defender_candidate_k_hop", 1),
                candidate_random_nodes=defender_cfg.get("defender_candidate_random_nodes", 0),
                candidate_high_risk_nodes=defender_cfg.get("defender_candidate_high_risk_nodes", 0),
                candidate_include_attack_target=defender_cfg.get("defender_candidate_include_attack_target", True),
                candidate_include_owned_neighbors=defender_cfg.get("defender_candidate_include_owned_neighbors", True),
                candidate_include_suspicious_nodes=defender_cfg.get("defender_candidate_include_suspicious_nodes", False),
                candidate_suspicion_threshold=defender_cfg.get("defender_candidate_suspicion_threshold", 0.35),
                attack_detection_mode=defender_cfg.get("defender_attack_detection_mode", "perfect"),
                attack_detection_base_prob=defender_cfg.get("defender_attack_detection_base_prob", 0.0),
                attack_detection_signal_gain=defender_cfg.get("defender_attack_detection_signal_gain", 0.0),
                attack_detection_age_gain=defender_cfg.get("defender_attack_detection_age_gain", 0.0),
                attack_detection_decay=defender_cfg.get("defender_attack_detection_decay", 1.0),
                attack_detection_success_bonus=defender_cfg.get("defender_attack_detection_success_bonus", 1.0),
                attack_detection_failure_bonus=defender_cfg.get("defender_attack_detection_failure_bonus", 1.0),
                attack_detection_max_prob=defender_cfg.get("defender_attack_detection_max_prob", 1.0),
                attack_detection_report_ttl=defender_cfg.get("defender_attack_detection_report_ttl", 1),
                attack_detection_event_bonus_scale=defender_cfg.get("defender_attack_detection_event_bonus_scale"),
                attack_detection_forced_age=defender_cfg.get("defender_attack_detection_forced_age", 0),
                attack_detection_force_after_missed_events=defender_cfg.get(
                    "defender_attack_detection_force_after_missed_events"
                ),
                reimage_requires_detection=defender_cfg.get("defender_reimage_requires_detection"),
                reimage_auto_focus_detected_node=defender_cfg.get(
                    "defender_reimage_auto_focus_detected_node", False
                ),
                blind_firewall_budget=defender_cfg.get("defender_blind_firewall_budget"),
                prioritize_reimage_over_service_actions=defender_cfg.get(
                    "defender_prioritize_reimage_over_service_actions", True
                ),
                log_episode_end=False,
                log_episode_summary=bool(defender_log_summary),
                episode_log_prefix=f"[env={idx}]",
            )
        )

    attacker_vec_env = DummyVecEnv([lambda w=w: w for w in attacker_wrappers])
    defender_vec_env = DummyVecEnv([lambda w=w: w for w in defender_wrappers])

    if config.get("norm_obs") or config.get("norm_reward"):
        attacker_vec_env = VecNormalize(
            attacker_vec_env, norm_obs=config.get("norm_obs", False), norm_reward=config.get("norm_reward", False)
        )

    return attacker_vec_env, defender_vec_env


def _build_sb3_logger(logs_folder: str, tb_dir: str, extra_app_log: Optional[str] = None) -> Logger:
    app_log_path = os.path.join(logs_folder, "app.log")
    os.makedirs(logs_folder, exist_ok=True)
    output_formats = [
        HumanOutputFormat(sys.stdout),
        HumanOutputFormat(open(app_log_path, "a")),
        CSVOutputFormat(os.path.join(logs_folder, "progress.csv")),
        TensorBoardOutputFormat(tb_dir),
    ]
    if extra_app_log:
        output_formats.append(HumanOutputFormat(open(extra_app_log, "a")))
    return Logger(logs_folder, output_formats)


def _validate_on_policy(algorithm: str):
    if algorithm not in ON_POLICY_ALGOS:
        raise ValueError(
            f"Algorithm '{algorithm}' is not supported for multi-agent training yet. "
            f"Supported (on-policy) algorithms: {sorted(ON_POLICY_ALGOS)}."
        )


def _build_model(algorithm: str, env, config: Dict, logs_folder: str, run_id: int, policy_kwargs: Dict, verbose: int, device: str):
    _validate_on_policy(algorithm)
    algo_config = copy.deepcopy(config)
    if algo_config.get("learning_rate_type") == "linear":
        learning_rate = linear_schedule(algo_config["learning_rate"], algo_config["learning_rate_final"])
    else:
        learning_rate = algo_config.get("learning_rate")

    for key in ["learning_rate_type", "learning_rate", "learning_rate_final"]:
        algo_config.pop(key, None)

    model_class = algorithm_models[algorithm]

    policy_kwargs = dict(policy_kwargs) if policy_kwargs else {}
    policy_override = policy_kwargs.pop("policy_class", None)
    if algorithm in reccurrent_algorithms:
        policy = policy_override or "MultiInputLstmPolicy"
    else:
        policy = policy_override or "MultiInputPolicy"
        for key in ["lstm_hidden_size", "n_lstm_layers"]:
            policy_kwargs.pop(key, None)

    return model_class(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        tensorboard_log=logs_folder,
        verbose=verbose,
        device=device,
        **algo_config,
    )


def _set_on_policy_n_steps(model, n_steps: int):
    """Align on-policy rollout length and re-create rollout buffer with the new size."""
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if not hasattr(model, "rollout_buffer_class"):
        return
    model.n_steps = n_steps
    rollout_kwargs = dict(getattr(model, "rollout_buffer_kwargs", {}) or {})
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **rollout_kwargs,
    )


def train_multiagent(
    logs_folder,
    envs_folder,
    config,
    train_ids,
    defender_cfg,
    logger=None,
    verbose=1,
    return_metrics=False,
):
    resume_enabled = bool(config.get("resume_from"))
    if verbose:
        logger.info(
            "Training attacker %s and defender %s on %d environments with %d runs.",
            config["algorithm"],
            defender_cfg["defender_algorithm"],
            config["num_environments"],
            config["num_runs"],
        )

    run_metrics = []
    for run_id in range(config["num_runs"]):
        seed = config["seeds_runs"][run_id]
        set_seeds(seed)
        if verbose:
            logger.info("Run %d/%d with seed %d started.", run_id + 1, config["num_runs"], seed)

        completion_flag = _run_completion_flag_path(logs_folder, run_id + 1)
        if resume_enabled and os.path.exists(completion_flag):
            if verbose:
                logger.info("Run %d already completed, skipping training.", run_id + 1)
            continue

        attacker_envs, defender_envs = _build_vec_envs(
            logs_folder, envs_folder, config, train_ids, seed, defender_cfg, logger=logger
        )
        # Persist runtime-expanded defender config (notably firewall_rule_list) so
        # evaluation can reuse exactly the same action-space semantics as training.
        try:
            save_yaml(defender_cfg, logs_folder, "defender_config.yaml")
        except Exception as exc:
            if logger:
                logger.warning("Failed to refresh runtime defender_config.yaml: %s", str(exc))

        attacker_cfg = config["algorithm_hyperparams"]
        defender_algo = defender_cfg.get("defender_algorithm", config["algorithm"])
        defender_algo_norm = str(defender_algo).lower()
        use_simple_defender = defender_algo_norm in {"noop", "random"}
        defender_cfg_algo = defender_cfg.get("defender_algorithm_hyperparams", attacker_cfg)

        attacker_policy_kwargs = config["policy_kwargs"]
        defender_policy_kwargs = defender_cfg.get("defender_policy_kwargs", attacker_policy_kwargs)
        defender_policy_name = str(defender_cfg.get("defender_policy", "")).lower()
        if defender_policy_name in {"factorized_node_attention", "factorized"}:
            defender_policy_kwargs = dict(defender_policy_kwargs) if defender_policy_kwargs else {}
            defender_policy_kwargs["policy_class"] = FactorizedDefenderPolicy
            defender_policy_kwargs["candidate_mask_mode"] = defender_cfg.get("defender_candidate_mask_mode", "soft")
            defender_policy_kwargs["node_prior_strength"] = float(defender_cfg.get("defender_node_prior_strength", 2.0))
            defender_policy_kwargs["port_prior_strength"] = float(defender_cfg.get("defender_port_prior_strength", 2.0))
            defender_policy_kwargs["service_prior_strength"] = float(defender_cfg.get("defender_service_prior_strength", 1.0))
            defender_policy_kwargs["condition_on_sampled_node"] = bool(defender_cfg.get("defender_condition_on_sampled_node", True))
            defender_policy_kwargs["condition_on_action_node"] = bool(defender_cfg.get("defender_condition_on_action_node", True))
            defender_policy_kwargs["canonical_node_index"] = int(defender_cfg.get("defender_canonical_node_index", 1))
            defender_policy_kwargs["enforce_node_consistency"] = bool(defender_cfg.get("defender_enforce_node_consistency", True))

        if defender_policy_kwargs is not attacker_policy_kwargs:
            defender_policy_kwargs = replace_with_classes(defender_policy_kwargs)

        device = config.get("device", "cpu")

        attacker_model = _build_model(
            config["algorithm"],
            attacker_envs,
            attacker_cfg,
            os.path.join(logs_folder, "attacker"),
            run_id + 1,
            attacker_policy_kwargs,
            verbose,
            device,
        )
        defender_model = None
        if not use_simple_defender:
            defender_model = _build_model(
                defender_algo,
                defender_envs,
                defender_cfg_algo,
                os.path.join(logs_folder, "defender"),
                run_id + 1,
                defender_policy_kwargs,
                verbose,
                device,
            )

        attacker_log_root = os.path.join(logs_folder, "attacker")
        defender_log_root = os.path.join(logs_folder, "defender")
        attacker_tb = _get_tensorboard_dir(attacker_log_root, config["algorithm"], run_id + 1)
        defender_tb = _get_tensorboard_dir(defender_log_root, defender_algo, run_id + 1)
        root_app_log = os.path.join(logs_folder, "app.log")
        attacker_logger = _build_sb3_logger(attacker_log_root, attacker_tb, extra_app_log=root_app_log)
        defender_logger = _build_sb3_logger(defender_log_root, defender_tb, extra_app_log=root_app_log)
        attacker_model.set_logger(attacker_logger)
        if defender_model is not None:
            defender_model.set_logger(defender_logger)

        attacker_agent = BaselineMultiAgent(attacker_model, logger, role="attacker")
        if use_simple_defender:
            defender_agent = SimpleMultiAgent(
                defender_envs,
                logger,
                role="defender",
                mode=defender_algo_norm,
                n_rollout_steps=attacker_model.n_steps,
                sb3_logger=defender_logger,
            )
            logger.info("Defender running in %s mode (no learning).", defender_algo_norm)
        else:
            defender_agent = BaselineMultiAgent(defender_model, logger, role="defender")

        if attacker_model.n_steps != defender_agent.n_rollout_steps:
            raise ValueError("Attacker and defender n_steps must be equal for synchronized rollouts.")

        attacker_ckpt_dir = os.path.join(logs_folder, "checkpoints", "attacker", str(run_id + 1))
        defender_ckpt_dir = os.path.join(logs_folder, "checkpoints", "defender", str(run_id + 1))
        os.makedirs(attacker_ckpt_dir, exist_ok=True)
        os.makedirs(defender_ckpt_dir, exist_ok=True)

        resume_loaded = False
        if resume_enabled:
            attacker_resume = _find_latest_checkpoint(attacker_ckpt_dir)
            defender_resume = None if use_simple_defender else _find_latest_checkpoint(defender_ckpt_dir)
            if attacker_resume:
                attacker_model = attacker_model.load(attacker_resume, env=attacker_envs, device=device)
                attacker_model.set_logger(attacker_logger)
                attacker_agent = BaselineMultiAgent(attacker_model, logger, role="attacker")
                logger.info("Resuming attacker from %s", attacker_resume)
                resume_loaded = True
            if defender_resume and defender_model is not None:
                defender_model = defender_model.load(defender_resume, env=defender_envs, device=device)
                defender_model.set_logger(defender_logger)
                defender_agent = BaselineMultiAgent(defender_model, logger, role="defender")
                logger.info("Resuming defender from %s", defender_resume)
                resume_loaded = True
        else:
            attacker_finetune = config.get("finetune_model")
            if attacker_finetune:
                if not os.path.exists(attacker_finetune):
                    raise ValueError(f"Attacker finetune model not found: {attacker_finetune}")
                attacker_model = attacker_model.load(attacker_finetune, env=attacker_envs, device=device)
                attacker_model.set_logger(attacker_logger)
                attacker_agent = BaselineMultiAgent(attacker_model, logger, role="attacker")
                logger.info("Loaded attacker pretrained model and continue training from %s", attacker_finetune)
            if not use_simple_defender:
                defender_finetune = defender_cfg.get("defender_finetune_model")
                if defender_finetune and os.path.exists(defender_finetune):
                    defender_model = defender_model.load(defender_finetune, env=defender_envs, device=device)
                    defender_model.set_logger(defender_logger)
                    defender_agent = BaselineMultiAgent(defender_model, logger, role="defender")
                    logger.info("Loaded defender finetune model from %s", defender_finetune)

        attacker_steps = int(attacker_model.n_steps)
        defender_steps = int(defender_agent.n_rollout_steps)
        if attacker_steps != defender_steps:
            aligned_steps = min(attacker_steps, defender_steps)
            logger.warning(
                "n_steps mismatch after model loading: attacker=%d defender=%d; aligning both to %d.",
                attacker_steps,
                defender_steps,
                aligned_steps,
            )
            _set_on_policy_n_steps(attacker_model, aligned_steps)
            if defender_model is not None and not use_simple_defender:
                _set_on_policy_n_steps(defender_model, aligned_steps)
            elif hasattr(defender_agent, "_n_rollout_steps"):
                defender_agent._n_rollout_steps = aligned_steps

        def checkpoint_saver(_iteration: int):
            step = min(attacker_agent.num_timesteps, defender_agent.num_timesteps)
            attacker_path = os.path.join(attacker_ckpt_dir, f"checkpoint_{step}_steps.zip")
            defender_path = os.path.join(defender_ckpt_dir, f"checkpoint_{step}_steps.zip")
            attacker_agent.save(attacker_path)
            defender_agent.save(defender_path)

        checkpoint_interval = int(config.get("marl_checkpoint_interval", 10))
        learn_multi(
            attacker_agent=attacker_agent,
            defender_agent=defender_agent,
            total_timesteps=config["train_iterations"],
            checkpoint_callback=checkpoint_saver,
            checkpoint_interval=max(1, checkpoint_interval),
            reset_num_timesteps=not resume_loaded,
            defender_first=bool(defender_cfg.get("defender_act_first", False)),
        )

        attacker_save_path = os.path.join(logs_folder, f"attacker_model_run_{run_id + 1}.zip")
        defender_save_path = os.path.join(logs_folder, f"defender_model_run_{run_id + 1}.zip")
        attacker_agent.save(attacker_save_path)
        defender_agent.save(defender_save_path)

        if return_metrics:
            run_metrics.append(
                {
                    "attacker": attacker_agent.get_episode_metrics(),
                    "defender": defender_agent.get_episode_metrics(),
                }
            )

        with open(_run_completion_flag_path(logs_folder, run_id + 1), "w") as f:
            f.write(f"completed at {datetime.now().isoformat()}\n")

    if return_metrics:
        return run_metrics
    return None


def setup_train_via_args_marl(args, logs_folder=None, envs_folder=None):
    args.yaml = args.yaml if "yaml" in args else False
    config_base_dir = script_dir
    original_cli_algorithm = str(args.algorithm).lower()

    attacker_model_cli = getattr(args, "attacker_model", None)
    finetune_model_cli = getattr(args, "finetune_model", None)
    if attacker_model_cli and finetune_model_cli and attacker_model_cli != finetune_model_cli:
        raise ValueError("Please provide only one of --attacker_model or --finetune_model, not both.")

    selected_attacker_model = attacker_model_cli or finetune_model_cli
    detected_attacker_algorithm = None
    if selected_attacker_model:
        resolved_attacker_model = _resolve_model_path(selected_attacker_model)
        detected_attacker_algorithm = _detect_on_policy_algorithm_from_model(resolved_attacker_model)
        if not detected_attacker_algorithm:
            raise ValueError(
                f"Unable to detect attacker algorithm from model: {resolved_attacker_model}. "
                "Expected one of: ppo, a2c, trpo, rppo."
            )
        if detected_attacker_algorithm not in ON_POLICY_ALGOS:
            raise ValueError(
                f"Detected attacker algorithm '{detected_attacker_algorithm}' is not supported in MARL loop. "
                f"Supported: {sorted(ON_POLICY_ALGOS)}"
            )
        args.algorithm = detected_attacker_algorithm
        args.attacker_model = resolved_attacker_model
        args.finetune_model = resolved_attacker_model

    if args.yaml:
        with open(os.path.join(script_dir, "logs", str(args.yaml), "train_config.yaml"), "r") as config_file:
            config = yaml.safe_load(config_file)
        args.goal = config["goal"]
        args.nlp_extractor = config["nlp_extractor"]
        if not detected_attacker_algorithm:
            args.algorithm = config["algorithm"]
        config.setdefault("save_csv_file", args.save_csv_file)
        config.setdefault("save_embeddings_csv_file", args.save_embeddings_csv_file)
        config.setdefault("verbose", args.verbose)
        config.setdefault("num_parallel_envs", args.num_parallel_envs)
        config.setdefault("env_sampling_mode", args.env_sampling_mode)
        config.setdefault("shuffle_train_envs", args.shuffle_train_envs)
        config.setdefault("max_train_envs", args.max_train_envs)
        config.setdefault("vec_env_type", args.vec_env_type)
        config.setdefault("resume_from", args.resume_from)

    if not logs_folder and args.resume_from:
        logs_folder = _resolve_resume_logs_folder(
            args.resume_from, args.algorithm, args.goal, args.nlp_extractor
        )

    if not logs_folder:
        if args.name:
            logs_folder = os.path.join(script_dir, "logs", args.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            logs_folder = os.path.join(script_dir, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        logs_folder = os.path.join(logs_folder, args.algorithm.upper() + "_x_" + args.goal + "_" + args.nlp_extractor)
        os.makedirs(logs_folder, exist_ok=True)

    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)
    if selected_attacker_model:
        logger.info("Using attacker pretrained model: %s", args.finetune_model)
        logger.info(
            "Auto-detected attacker algorithm '%s' (CLI requested '%s').",
            args.algorithm,
            original_cli_algorithm,
        )
    logger.info("Command-line args: %s", vars(args))
    save_yaml(vars(args), logs_folder, "cli_args.yaml")

    if "yaml" in args and args.yaml:
        config["yaml"] = args.yaml
        if detected_attacker_algorithm:
            with open(os.path.join(config_base_dir, args.algo_config), "r") as config_file:
                algorithm_config = yaml.safe_load(config_file)
            config["algorithm"] = detected_attacker_algorithm
            config["algorithm_hyperparams"] = algorithm_config[detected_attacker_algorithm]
            config["policy_kwargs"] = algorithm_config["policy_kwargs"]
            config["finetune_model"] = args.finetune_model
    else:
        if args.verbose:
            logger.info("Reading default configuration YAML files")
        with open(os.path.join(config_base_dir, args.train_config), "r") as config_file:
            config = yaml.safe_load(config_file)
        with open(os.path.join(config_base_dir, args.rewards_config), "r") as config_file:
            rewards_config = yaml.safe_load(config_file)
        with open(os.path.join(config_base_dir, args.algo_config), "r") as config_file:
            algorithm_config = yaml.safe_load(config_file)
        args.algorithm_hyperparams = algorithm_config[args.algorithm]
        args.rewards_dict = rewards_config["rewards_dict"][args.goal]
        args.penalties_dict = rewards_config["penalties_dict"][args.goal]
        args.policy_kwargs = algorithm_config["policy_kwargs"]
        config.update(vars(args))

    if config.get("proportional_cutoff_coefficient") is None:
        config["proportional_cutoff_coefficient"] = 0
    if config["proportional_cutoff_coefficient"] <= 0:
        if config.get("verbose"):
            logger.info("proportional_cutoff_coefficient <= 0 detected; disabling proportional cutoff.")
        config["proportional_cutoff_coefficient"] = 0

    if getattr(args, "finetune_model", None):
        resolved_finetune = _resolve_model_path(args.finetune_model)
        if ".zip" not in resolved_finetune.lower():
            raise ValueError("The path provided should be the zip file related to the model to finetune!")
        config["finetune_model"] = resolved_finetune

    if config["static_seeds"]:
        if config["verbose"]:
            logger.info("Setting a static seed for all runs of training")
        seeds_runs = [42 for _ in range(config["num_runs"])]
    elif config["random_seeds"]:
        if config["verbose"]:
            logger.info("Setting random seeds for training")
        seeds_runs = [np.random.randint(1000) for _ in range(config["num_runs"])]
    else:
        if config["verbose"]:
            logger.info("Loading seeds from seeds file %s", config["load_seeds"])
        seeds_loaded = load_yaml(os.path.join(config_base_dir, args.load_seeds, "seeds.yaml"))
        seeds_runs = seeds_loaded["seeds"][0:config["num_runs"]]

    save_yaml({"seeds": seeds_runs}, logs_folder, "seeds.yaml")
    config.update({"seeds_runs": seeds_runs})
    set_seeds(seeds_runs[0])

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    def get_base_path(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        run_dirs = sorted([d for d in subdirs if re.match(r"run_\\d+", d)])
        if run_dirs:
            return os.path.join(base_dir, run_dirs[-1])
        return base_dir

    if config["pca_components"]:
        if config["verbose"]:
            logger.info(
                "Loading graph encoder model related to NLP extractor %s with dimensionality reduction components %d",
                config["nlp_extractor"], config["pca_components"]
            )
        if config["yaml"]:
            graph_encoder_path = config["graph_encoder_path"]
            graph_encoder_config_path = config["graph_encoder_config_path"]
            graph_encoder_spec_path = config["graph_encoder_spec_path"]
        else:
            general_config = load_yaml(os.path.join(project_root, "config.yaml"))
            base_dir = os.path.join(
                cyberbattle_root,
                "gae",
                "logs",
                general_config["gae_dimensionality_reduction_path"][config["pca_components"]],
                f"num_components={config['pca_components']}",
                config["nlp_extractor"],
            )
            base_dir = get_base_path(base_dir)
            graph_encoder_path = os.path.join(base_dir, "encoder.pth")
            graph_encoder_config_path = os.path.join(base_dir, "train_config_encoder.yaml")
            graph_encoder_spec_path = os.path.join(base_dir, "model_spec.yaml")
    else:
        if config["verbose"]:
            logger.info("Loading graph encoder model related to NLP extractor %s", config["nlp_extractor"])
        if config["yaml"]:
            graph_encoder_path = config["graph_encoder_path"]
            graph_encoder_config_path = config["graph_encoder_config_path"]
            graph_encoder_spec_path = config["graph_encoder_spec_path"]
        else:
            general_config = load_yaml(os.path.join(project_root, "config.yaml"))
            base_dir = os.path.join(
                cyberbattle_root,
                "gae",
                "logs",
                general_config["gae_path"],
                config["nlp_extractor"],
            )
            base_dir = get_base_path(base_dir)
            graph_encoder_path = os.path.join(base_dir, "encoder.pth")
            graph_encoder_spec_path = os.path.join(base_dir, "model_spec.yaml")
            graph_encoder_config_path = os.path.join(base_dir, "train_config_encoder.yaml")

    if not config["pca_components"]:
        config.update({"pca_components": config["default_vulnerability_embeddings_size"]})

    with open(graph_encoder_config_path, "r") as config_file:
        config_encoder = yaml.safe_load(config_file)
    with open(graph_encoder_spec_path, "r") as config_file:
        spec_encoder = yaml.safe_load(config_file)
    config_encoder.update(spec_encoder)
    config["node_embeddings_dimensions"] = config_encoder["model_config"]["layers"][-1]["out_channels"]
    config["edge_feature_aggregations"] = config_encoder["edge_feature_aggregations"]

    device = config["device"]
    graph_encoder = GAEEncoder(
        config_encoder["node_feature_vector_size"],
        config_encoder["model_config"]["layers"],
        config_encoder["edge_feature_vector_size"],
    )
    graph_encoder.load_state_dict(torch.load(str(graph_encoder_path), map_location=device))
    graph_encoder = graph_encoder.to(device)
    graph_encoder.eval()

    config.update({"graph_encoder_path": graph_encoder_path})
    config.update({"graph_encoder_config_path": graph_encoder_config_path})
    config.update({"graph_encoder_spec_path": graph_encoder_spec_path})

    config = clean_config_save(config)
    save_yaml(config, logs_folder)

    config.update({"graph_encoder_path": os.path.join(script_dir, graph_encoder_path)})
    config.update({"graph_encoder_config_path": os.path.join(script_dir, graph_encoder_config_path)})
    config.update({"graph_encoder_spec_path": os.path.join(script_dir, graph_encoder_spec_path)})

    if config["load_processed_envs"]:
        if "/" not in config["load_processed_envs"]:
            raise ValueError("The path of the run folder inside the logs folder should be provided!")
        if config["verbose"]:
            logger.info("Loading processed environments from the run folder %s", config["load_processed_envs"])
        envs_folder = os.path.join(script_dir, "logs", config["load_processed_envs"], "envs")
        config["num_environments"] = 0
        for file in os.listdir(envs_folder):
            if file.endswith(".pkl"):
                config["num_environments"] += 1
        if config["verbose"]:
            logger.info("Loaded %d environments from the folder", config["num_environments"])
    elif config["load_envs"]:
        if config["verbose"]:
            logger.info("Loading environments from the folder %s", config["load_envs"])

        original_envs_folder = os.path.join(cyberbattle_root, "data", "env_samples", config["load_envs"])

        if not envs_folder:
            envs_folder = os.path.join(logs_folder, "envs")
        if not os.path.exists(os.path.join(envs_folder)):
            os.makedirs(os.path.join(envs_folder))

        config["num_environments"] = 0
        for element in tqdm(os.listdir(original_envs_folder), desc="Loading environments from the folder"):
            if os.path.isfile(os.path.join(original_envs_folder, element)):
                with open(os.path.join(envs_folder, element), "wb") as f:
                    shutil.copyfile(os.path.join(original_envs_folder, element), os.path.join(envs_folder, element))
            if os.path.isdir(os.path.join(original_envs_folder, element)) and element.isdigit():
                config["num_environments"] += 1

                if config["pca_components"] == 768:
                    network_folder = os.path.join(
                        original_envs_folder, element, f"network_{config['nlp_extractor']}.pkl"
                    )
                else:
                    network_folder = os.path.join(
                        original_envs_folder,
                        element,
                        "pca/num_components=" + str(config["pca_components"]),
                        f"network_{config['nlp_extractor']}.pkl",
                    )
                with open(network_folder, "rb") as f:
                    network = pickle.load(f)

                if config["static_defender_agent"]:
                    if config["static_defender_agent"] == "reimage":
                        config["static_defender_agent"] = ScanAndReimageCompromisedMachines(
                            random.uniform(config["detect_probability_min"], config["detect_probability_max"]),
                            random.randint(config["scan_capacity_min"], config["scan_capacity_max"]),
                            random.randint(config["scan_frequency_min"], config["scan_frequency_max"]),
                            logger=logger,
                            verbose=config["verbose"],
                        )
                    elif config["static_defender_agent"] == "events":
                        config["static_defender_agent"] = ExternalRandomEvents(
                            random.uniform(config["random_event_probability_min"], config["random_event_probability_max"]),
                            logger=logger,
                            verbose=config["verbose"],
                        )

                if args.environment_type == "global":
                    env = wrap_graphs_to_global_envs(network, logger, **config)
                elif args.environment_type == "local":
                    env = wrap_graphs_to_local_envs(network, logger, **config)
                else:
                    env = wrap_graphs_to_compressed_envs(network, logger, **config)
                    env.set_graph_encoder(graph_encoder)

                env.set_pca_components(config["pca_components"])

                with open(os.path.join(envs_folder, f"{element}.pkl"), "wb") as f:
                    pickle.dump(env, f)

        if config["verbose"]:
            logger.info("Loaded %d environments from the folder", config["num_environments"])

    if config["holdout"]:
        yaml_split_path = os.path.join(envs_folder, "split.yaml")
        with open(yaml_split_path, "r") as file:
            yaml_info = yaml.safe_load(file)
        train_ids = [elem["id"] for elem in yaml_info["training_set"]]
        val_ids = [elem["id"] for elem in yaml_info["validation_set"]]
    else:
        train_ids = [i + 1 for i in range(config["num_environments"])]
        val_ids = None
        yaml_info = {"training_set": [{"id": i} for i in train_ids], "validation_set": []}

    if config.get("max_train_envs"):
        if config["shuffle_train_envs"]:
            random.Random(seeds_runs[0]).shuffle(train_ids)
        train_ids = train_ids[:config["max_train_envs"]]
    config["num_train_envs"] = len(train_ids)
    config["train_env_ids"] = train_ids
    config["val_env_ids"] = val_ids or []

    with open(os.path.join(logs_folder, "split.yaml"), "w") as f:
        yaml.dump(yaml_info, f)

    save_yaml(config, logs_folder)
    config["policy_kwargs"] = replace_with_classes(config["policy_kwargs"])

    return logger, logs_folder, envs_folder, config, train_ids, val_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train attacker+defender on multiple C-CyberBattleSim environments")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "a2c", "rppo", "trpo", "ddpg", "sac", "td3", "tqc"], default="trpo")
    parser.add_argument("--environment_type", type=str, choices=["continuous", "global", "local"], default="continuous")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("-nlp", "--nlp_extractor", type=str, choices=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT", "SecRoBERTa"], default="CySecBERT")
    parser.add_argument("--holdout", action="store_true", default=True)
    parser.add_argument("--attacker_model", type=str, default=None,
                        help="Path to a pretrained attacker model (.zip). Algorithm is auto-detected and overrides --algorithm.")
    parser.add_argument("--finetune_model", type=str, help="Backward-compatible alias of --attacker_model")
    parser.add_argument("--early_stopping", type=int, default=7)
    parser.add_argument("-g", "--goal", type=str, default="control", choices=["control", "discovery", "disruption", "control_node", "discovery_node", "disruption_node"])
    parser.add_argument("--yaml", default=False, help="Read configuration file from YAML file of a previous training")
    parser.add_argument("--name", default="ALGO_MULTI_MARL", help="Name of the logs folder related to the run")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to an existing logs folder to resume from (relative to logs/ or absolute)")
    parser.add_argument("--load_envs", default="syntethic_deployment_20_graphs_100_nodes", help="Path of the envs folder where the networks should be processed and loaded from")
    parser.add_argument("--load_processed_envs", default=False, help="Path of the run folder where the envs already processed should be loaded from")
    parser.add_argument("--static_seeds", action="store_true", default=True)
    parser.add_argument("--load_seeds", default="config", help="Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)")
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

    parser.add_argument("--num_parallel_envs", type=int, default=4, help="Number of environments to run in parallel")
    parser.add_argument("--env_sampling_mode", choices=["partitioned", "random"], default="partitioned", help="How to assign env IDs across workers")
    parser.add_argument("--shuffle_train_envs", action="store_true", default=True, help="Shuffle env IDs before assigning to workers")
    parser.add_argument("--max_train_envs", type=int, default=8, help="Limit number of training environments (n)")
    parser.add_argument("--vec_env_type", choices=["dummy"], default="dummy")
    parser.add_argument("--marl_checkpoint_interval", type=int, default=10, help="Save attacker+defender checkpoints every N training iterations")

    # Defender-specific options
    parser.add_argument("--defender_config", type=str, default="config/defender_config.yaml", help="Path to defender config YAML")
    parser.add_argument("--defender_algorithm", type=str, default="trpo",
                        help="Override defender algorithm (ppo/a2c/trpo or noop/random)")
    parser.add_argument("--defender_algo_config", type=str, default="config/defender_algo_config.yaml",
                        help="Override defender algorithm config YAML")
    parser.add_argument("--defender_finetune_model", type=str, default=None, help="Path to defender model to finetune (relative to logs folder)")

    args = parser.parse_args()

    error, message = check_args(args)
    if error:
        raise ValueError(message)

    general_config = load_yaml(os.path.join(project_root, "config.yaml"))
    if not args.load_envs and not args.load_processed_envs:
        args.load_envs = general_config["default_environments_path"]

    logger, logs_folder, envs_folder, config, train_ids, _val_ids = setup_train_via_args_marl(args, None)
    config["marl_checkpoint_interval"] = args.marl_checkpoint_interval

    if config.get("static_defender_agent"):
        logger.warning("Static defender agent is set; disabling it for MARL training.")
        config["static_defender_agent"] = None

    defender_cfg_path = args.defender_config
    if args.yaml:
        candidate = os.path.join(logs_folder, "defender_config.yaml")
        if os.path.exists(candidate):
            defender_cfg_path = candidate
    if not os.path.isabs(defender_cfg_path):
        defender_cfg_path = os.path.join(script_dir, defender_cfg_path)
    with open(defender_cfg_path, "r") as f:
        defender_cfg = yaml.safe_load(f) or {}

    if args.defender_algorithm:
        defender_cfg["defender_algorithm"] = args.defender_algorithm

    if args.defender_algo_config:
        algo_path = args.defender_algo_config
        if not os.path.isabs(algo_path):
            algo_path = os.path.join(script_dir, algo_path)
        with open(algo_path, "r") as f:
            algo_cfg = yaml.safe_load(f) or {}
        # Expect same structure as algo_config.yaml
        defender_algo_name = defender_cfg.get("defender_algorithm", config["algorithm"])
        defender_cfg["defender_algorithm_hyperparams"] = algo_cfg.get(defender_algo_name, algo_cfg)
        if "policy_kwargs" in algo_cfg:
            defender_cfg["defender_policy_kwargs"] = algo_cfg["policy_kwargs"]
    else:
        defender_cfg["defender_algorithm_hyperparams"] = config["algorithm_hyperparams"]

    if args.defender_finetune_model:
        defender_cfg["defender_finetune_model"] = args.defender_finetune_model

    # Save defender config alongside training config
    save_yaml(defender_cfg, logs_folder, "defender_config.yaml")

    train_multiagent(logs_folder, envs_folder, config, train_ids, defender_cfg, logger=logger, verbose=args.verbose)
