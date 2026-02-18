from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
cyberbattle_root = os.path.join(project_root, "cyberbattle")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cyberbattle.utils.train_utils import algorithm_models  # noqa: E402
from cyberbattle.utils.math_utils import set_seeds  # noqa: E402
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv  # noqa: E402
from cyberbattle.agents.multi_env_marl.env_wrappers.environment_event_source import (  # noqa: E402
    EnvironmentEventSource,
)
from cyberbattle.agents.multi_env_marl.env_wrappers.shared_env import SharedResetEnv  # noqa: E402
from cyberbattle.agents.multi_env_marl.env_wrappers.attacker_wrapper import AttackerEnvWrapper  # noqa: E402
from cyberbattle.agents.multi_env_marl.env_wrappers.defender_wrapper import DefenderEnvWrapper  # noqa: E402
from cyberbattle.agents.multi_env_marl.policies import FactorizedDefenderPolicy  # noqa: F401,E402


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_bool_mode(mode: str, fallback: bool) -> bool:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "auto":
        return bool(fallback)
    if mode_norm == "true":
        return True
    if mode_norm == "false":
        return False
    raise ValueError(f"Invalid boolean mode '{mode}', expected one of: auto/true/false")


def _setup_eval_logger(
    run_folder: str,
    verbose: int,
    eval_log_path: Optional[str],
    sync_to_run_app_log: bool,
) -> logging.Logger:
    level = logging.INFO if int(verbose) > 0 else logging.WARNING
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    handlers: List[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if eval_log_path:
        eval_log_path = os.path.expanduser(eval_log_path)
        if not os.path.isabs(eval_log_path):
            eval_log_path = os.path.join(run_folder, eval_log_path)
        eval_log_path = os.path.abspath(eval_log_path)
    else:
        eval_log_path = os.path.join(run_folder, "evaluation", "app.log")
    os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)
    eval_file_handler = logging.FileHandler(eval_log_path, mode="a", encoding="utf-8")
    eval_file_handler.setLevel(level)
    eval_file_handler.setFormatter(formatter)
    handlers.append(eval_file_handler)

    if sync_to_run_app_log:
        run_app_log = os.path.join(run_folder, "app.log")
        run_handler = logging.FileHandler(run_app_log, mode="a", encoding="utf-8")
        run_handler.setLevel(level)
        run_handler.setFormatter(formatter)
        handlers.append(run_handler)

    managed_loggers = [
        logging.getLogger("marl.eval"),
        logging.getLogger("marl.episode"),
        logging.getLogger("cyberbattle.attacker"),
        logging.getLogger("cyberbattle.defender"),
    ]
    for lg in managed_loggers:
        lg.setLevel(level)
        lg.propagate = False
        for old_handler in list(lg.handlers):
            lg.removeHandler(old_handler)
        for handler in handlers:
            lg.addHandler(handler)

    return logging.getLogger("marl.eval")


def _parse_ids(raw: str) -> List[int]:
    if not raw:
        return []
    parts = re.split(r"[\s,]+", raw.strip())
    ids = []
    for part in parts:
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"Invalid environment id '{part}', expected integer.")
        ids.append(int(part))
    return sorted(set(ids))


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_run_folder(run_folder: str) -> str:
    candidate = os.path.expanduser(str(run_folder).strip())
    attempts = []
    if os.path.isabs(candidate):
        attempts.append(candidate)
    else:
        attempts.extend(
            [
                candidate,
                os.path.join(script_dir, candidate),
                os.path.join(script_dir, "logs", candidate),
                os.path.join(os.getcwd(), candidate),
            ]
        )

    for path in attempts:
        path = os.path.abspath(path)
        if os.path.isfile(path) and os.path.basename(path).lower().endswith(".log"):
            path = os.path.dirname(path)
        if os.path.isdir(path):
            return path
    attempted = "\n".join(os.path.abspath(p) for p in attempts)
    raise ValueError(f"Run folder not found: {run_folder}\nTried:\n{attempted}")


def _resolve_model_path(model_path: str, run_folder: str) -> str:
    candidate = os.path.expanduser(str(model_path).strip())
    attempts = []
    if os.path.isabs(candidate):
        attempts.append(candidate)
    else:
        attempts.extend(
            [
                candidate,
                os.path.join(run_folder, candidate),
                os.path.join(script_dir, candidate),
                os.path.join(script_dir, "logs", candidate),
                os.path.join(os.getcwd(), candidate),
            ]
        )

    for path in attempts:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            return path
    attempted = "\n".join(os.path.abspath(p) for p in attempts)
    raise ValueError(f"Model path not found: {model_path}\nTried:\n{attempted}")


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
    match = re.search(r"(?i)(?:^|[/_\-])(rppo|trpo|ppo|a2c|ddpg|sac|td3|tqc|dqn)(?:[/_\-]|$)", normalized)
    if match:
        return match.group(1).lower()
    return None


def _detect_algorithm_from_model(model_path: str) -> Optional[str]:
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
        if {"buffer_size", "tau", "train_freq"} & keys and "action_noise" in keys:
            return "ddpg"
        if {"ent_coef", "target_entropy"} & keys:
            return "sac"
        if {"policy_delay", "target_policy_noise"} & keys:
            return "td3"
        if {"top_quantiles_to_drop_per_net", "n_quantiles"} & keys:
            return "tqc"
        if {"exploration_fraction", "target_update_interval"} & keys:
            return "dqn"
    return _detect_algorithm_from_path(model_path)


def _load_sb3_model(model_path: str, algorithm: Optional[str], device: str):
    algo = str(algorithm).lower() if algorithm else _detect_algorithm_from_model(model_path)
    if not algo:
        raise ValueError(f"Unable to detect algorithm from model: {model_path}")
    if algo not in algorithm_models:
        raise ValueError(f"Unsupported algorithm '{algo}' for model {model_path}")
    cls = algorithm_models[algo]
    model = cls.load(model_path, device=device)
    return algo, model


def _list_env_ids_from_folder(envs_folder: str) -> List[int]:
    env_ids = []
    for filename in os.listdir(envs_folder):
        if not filename.endswith(".pkl"):
            continue
        stem = filename[:-4]
        if stem.isdigit():
            env_ids.append(int(stem))
    return sorted(env_ids)


def _load_split_ids(split_yaml_path: str, key: str) -> List[int]:
    if not os.path.exists(split_yaml_path):
        return []
    data = _read_yaml(split_yaml_path)
    values = data.get(key, [])
    ids: List[int] = []
    for item in values:
        if isinstance(item, dict) and "id" in item:
            try:
                ids.append(int(item["id"]))
            except Exception:
                continue
        elif isinstance(item, int):
            ids.append(int(item))
    return sorted(set(ids))


def _scan_defender_bounds(
    envs_folder: str,
    env_ids: Sequence[int],
    firewall_rule_limit: Optional[int],
    firewall_rule_source: str,
    service_action_limit: Optional[int],
    firewall_rule_min_support: Optional[int],
) -> Tuple[int, int, List[str], float]:
    max_nodes = 1
    max_total_services = 1
    port_counts: Dict[str, int] = {}
    total_port_events = 0

    source = str(firewall_rule_source or "vulnerabilities").lower()
    service_limit = int(service_action_limit) if service_action_limit is not None else None
    min_support = int(firewall_rule_min_support) if firewall_rule_min_support is not None else None

    for env_id in env_ids:
        path = os.path.join(envs_folder, f"{env_id}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            env = pickle.load(f)

        nodes = list(env.environment.nodes)
        max_nodes = max(max_nodes, len(nodes))
        total_services = 0

        for node_id in nodes:
            node_info = env.get_node(node_id)
            total_services += len(node_info.services)

            if source in {"services", "both"}:
                for idx, svc in enumerate(node_info.services):
                    if service_limit is not None and idx >= service_limit:
                        break
                    port = svc.name
                    port_counts[port] = port_counts.get(port, 0) + 1
                    total_port_events += 1

            if source in {"vulnerabilities", "both"}:
                for vuln in node_info.vulnerabilities.values():
                    port = vuln.port
                    port_counts[port] = port_counts.get(port, 0) + 1
                    total_port_events += 1

        max_total_services = max(max_total_services, total_services)

    ranked_ports = sorted(port_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    if min_support is not None and min_support > 1:
        ranked_ports = [item for item in ranked_ports if item[1] >= min_support]

    if firewall_rule_limit is not None and int(firewall_rule_limit) > 0:
        ranked_ports = ranked_ports[: int(firewall_rule_limit)]

    firewall_ports = [port for port, _count in ranked_ports]
    covered = sum(port_counts.get(p, 0) for p in firewall_ports)
    coverage = float(covered) / float(total_port_events) if total_port_events else 0.0

    return max_nodes, max_total_services, firewall_ports, coverage


def _pick_env_ids(
    train_cfg: Dict[str, Any],
    run_folder: str,
    envs_folder: str,
    env_split: str,
    explicit_env_ids: Optional[str],
    logger: logging.Logger,
) -> List[int]:
    if explicit_env_ids:
        ids = _parse_ids(explicit_env_ids)
        if not ids:
            raise ValueError("--env_ids was provided but no valid IDs were parsed.")
        return ids

    split_yaml = os.path.join(run_folder, "split.yaml")
    train_ids = _load_split_ids(split_yaml, "training_set")
    val_ids = _load_split_ids(split_yaml, "validation_set")

    if not train_ids:
        train_ids = [int(i) for i in train_cfg.get("train_env_ids", [])]

    if env_split == "train":
        ids = train_ids
    elif env_split == "val":
        ids = val_ids
        if not ids:
            logger.warning("Validation split empty, falling back to training split.")
            ids = train_ids
    else:
        ids = sorted(set(train_ids + val_ids))

    if not ids:
        ids = _list_env_ids_from_folder(envs_folder)

    if not ids:
        raise ValueError(f"No environment IDs found in {envs_folder}")

    return ids


def _build_defender_bounds(
    envs_folder: str,
    eval_env_ids: Sequence[int],
    train_env_ids_for_ports: Sequence[int],
    defender_cfg: Dict[str, Any],
    defender_model: Optional[Any],
    logger: logging.Logger,
) -> Tuple[int, int, int, List[str], float]:
    service_action_limit = int(defender_cfg.get("defender_service_action_limit", 3))
    fw_limit = defender_cfg.get("defender_firewall_rule_limit")
    fw_source = defender_cfg.get("defender_firewall_rule_source", "vulnerabilities")
    fw_min_support = defender_cfg.get("defender_firewall_rule_min_support")

    max_nodes_eval, max_total_services_eval, firewall_ports_eval, coverage_eval = _scan_defender_bounds(
        envs_folder,
        eval_env_ids,
        firewall_rule_limit=fw_limit,
        firewall_rule_source=fw_source,
        service_action_limit=service_action_limit,
        firewall_rule_min_support=fw_min_support,
    )

    firewall_ports = list(firewall_ports_eval)
    max_nodes = max_nodes_eval
    max_total_services = max_total_services_eval

    if defender_model is not None:
        nvec = getattr(getattr(defender_model, "action_space", None), "nvec", None)
        if nvec is None or len(nvec) < 10:
            raise ValueError("Loaded defender model does not expose expected MultiDiscrete action_space.nvec")

        expected_nodes = int(nvec[1])
        expected_ports = int(nvec[3])
        expected_services = int(nvec[9])

        # Use training IDs as primary source for defender port vocabulary to keep semantics stable.
        max_nodes_train, max_total_services_train, firewall_ports_train, _cov_train = _scan_defender_bounds(
            envs_folder,
            train_env_ids_for_ports if train_env_ids_for_ports else eval_env_ids,
            firewall_rule_limit=fw_limit,
            firewall_rule_source=fw_source,
            service_action_limit=expected_services,
            firewall_rule_min_support=fw_min_support,
        )

        max_nodes = expected_nodes
        max_total_services = max(max_total_services_train, max_total_services_eval, 1)
        service_action_limit = expected_services

        configured_list = defender_cfg.get("defender_firewall_rule_list")
        if isinstance(configured_list, list):
            if configured_list:
                firewall_ports = [str(p) for p in configured_list]
            else:
                # Keep compatibility with training wrapper semantics:
                # empty list means DefenderEnvWrapper falls back to class defaults.
                firewall_ports = list(DefenderEnvWrapper.firewall_rule_list)
                logger.info(
                    "defender_firewall_rule_list is empty in config; using wrapper default ports %s",
                    firewall_ports,
                )
        elif firewall_ports_train:
            firewall_ports = list(firewall_ports_train)

        if len(firewall_ports) == 0 and expected_ports > 0:
            _, _, relaxed_ports, _ = _scan_defender_bounds(
                envs_folder,
                train_env_ids_for_ports if train_env_ids_for_ports else eval_env_ids,
                firewall_rule_limit=fw_limit,
                firewall_rule_source=fw_source,
                service_action_limit=expected_services,
                firewall_rule_min_support=1,
            )
            if relaxed_ports:
                firewall_ports = list(relaxed_ports)
                logger.warning(
                    "Firewall port list empty with min_support=%s; fallback to min_support=1 recovered %d ports.",
                    str(fw_min_support),
                    len(firewall_ports),
                )
            elif str(fw_source).lower() != "both":
                _, _, relaxed_ports_both, _ = _scan_defender_bounds(
                    envs_folder,
                    train_env_ids_for_ports if train_env_ids_for_ports else eval_env_ids,
                    firewall_rule_limit=fw_limit,
                    firewall_rule_source="both",
                    service_action_limit=expected_services,
                    firewall_rule_min_support=1,
                )
                if relaxed_ports_both:
                    firewall_ports = list(relaxed_ports_both)
                    logger.warning(
                        "Firewall port list still empty; fallback source=both,min_support=1 recovered %d ports.",
                        len(firewall_ports),
                    )

        if len(firewall_ports) < expected_ports:
            logger.warning(
                "Firewall port list shorter than defender model n_ports (%d < %d). Padding placeholder ports.",
                len(firewall_ports),
                expected_ports,
            )
        firewall_ports = firewall_ports[:expected_ports]
        while len(firewall_ports) < expected_ports:
            firewall_ports.append(f"__pad_port_{len(firewall_ports)}")

        logger.info(
            "Defender model space constraints applied: max_nodes=%d n_ports=%d service_action_limit=%d",
            max_nodes,
            expected_ports,
            service_action_limit,
        )

    if not firewall_ports:
        firewall_ports = ["RDP"]

    logger.info(
        "Defender bounds: max_nodes=%d max_total_services=%d service_action_limit=%d firewall_ports=%d coverage=%.3f",
        max_nodes,
        max_total_services,
        service_action_limit,
        len(firewall_ports),
        coverage_eval,
    )

    return max_nodes, max_total_services, service_action_limit, firewall_ports, coverage_eval


def _get_wrapper_reason(wrapper: Any) -> str:
    if hasattr(wrapper, "_infer_episode_reason"):
        try:
            return str(wrapper._infer_episode_reason())
        except Exception:
            pass
    if getattr(wrapper, "last_outcome", None):
        return str(wrapper.last_outcome)
    return "unknown"


def _decide_winner(attacker_reason: str, defender_reason: str) -> str:
    if attacker_reason == "attacker_win":
        return "attacker"
    defender_win_reasons = {
        "defender_win",
        "attacker_lost",
        "sla_breached",
    }
    if attacker_reason in defender_win_reasons or defender_reason in defender_win_reasons:
        return "defender"
    return "other"


def _normalize_model_action(action: Any) -> Any:
    if isinstance(action, np.ndarray) and action.ndim > 1:
        return action[0]
    return action


def _choose_defender_action(
    mode: str,
    defender_model: Optional[Any],
    defender_obs: Any,
    defender_env: DefenderEnvWrapper,
    deterministic: bool,
) -> Any:
    if mode == "noop":
        return []
    if mode == "random":
        return defender_env.action_space.sample()
    if defender_model is None:
        raise ValueError("defender_mode='model' requires a loaded defender model")
    action, _ = defender_model.predict(defender_obs, deterministic=deterministic)
    return _normalize_model_action(action)


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    run_folder = _resolve_run_folder(args.run_folder)
    logs_folder = os.path.join(run_folder, "evaluation")
    os.makedirs(logs_folder, exist_ok=True)
    resolved_eval_log_path = (
        os.path.abspath(os.path.join(run_folder, os.path.expanduser(args.eval_log_path)))
        if args.eval_log_path and not os.path.isabs(os.path.expanduser(args.eval_log_path))
        else (
            os.path.abspath(os.path.expanduser(args.eval_log_path))
            if args.eval_log_path
            else os.path.join(run_folder, "evaluation", "app.log")
        )
    )
    logger = _setup_eval_logger(
        run_folder=run_folder,
        verbose=int(args.verbose),
        eval_log_path=args.eval_log_path,
        sync_to_run_app_log=_as_bool(args.sync_to_run_app_log, True),
    )

    train_cfg_path = os.path.join(run_folder, "train_config.yaml")
    if not os.path.exists(train_cfg_path):
        raise ValueError(f"train_config.yaml not found in run folder: {run_folder}")
    train_cfg = _read_yaml(train_cfg_path)

    defender_cfg_path = args.defender_config or os.path.join(run_folder, "defender_config.yaml")
    if not os.path.isabs(defender_cfg_path):
        defender_cfg_path = os.path.abspath(os.path.join(run_folder, defender_cfg_path))
    if not os.path.exists(defender_cfg_path):
        fallback = os.path.join(script_dir, "config", "defender_config.yaml")
        logger.warning("defender_config not found at %s, fallback to %s", defender_cfg_path, fallback)
        defender_cfg_path = fallback
    defender_cfg = _read_yaml(defender_cfg_path)

    envs_folder = args.envs_folder
    if not envs_folder:
        envs_folder = os.path.join(run_folder, "envs")
    if not os.path.isabs(envs_folder):
        envs_folder = os.path.abspath(os.path.join(run_folder, envs_folder))
    if not os.path.isdir(envs_folder):
        raise ValueError(f"Environments folder not found: {envs_folder}")

    eval_env_ids = _pick_env_ids(
        train_cfg,
        run_folder,
        envs_folder,
        env_split=args.env_split,
        explicit_env_ids=args.env_ids,
        logger=logger,
    )

    train_env_ids = [int(i) for i in train_cfg.get("train_env_ids", [])]
    if not train_env_ids:
        train_env_ids = list(eval_env_ids)

    run_id = int(args.run_id)
    attacker_model_path = args.attacker_model or os.path.join(run_folder, f"attacker_model_run_{run_id}.zip")
    attacker_model_path = _resolve_model_path(attacker_model_path, run_folder)

    defender_mode = str(args.defender_mode).lower()
    defender_model_path = None
    if defender_mode == "model":
        defender_model_path = args.defender_model or os.path.join(run_folder, f"defender_model_run_{run_id}.zip")
        defender_model_path = _resolve_model_path(defender_model_path, run_folder)

    attacker_log_episode_summary = _resolve_bool_mode(
        args.attacker_log_episode_summary,
        _as_bool(train_cfg.get("attacker_log_episode_summary", False), False),
    )
    defender_log_episode_summary = _resolve_bool_mode(
        args.defender_log_episode_summary,
        _as_bool(defender_cfg.get("defender_log_episode_summary", False), False),
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seeds(int(args.seed))

    attacker_algo, attacker_model = _load_sb3_model(attacker_model_path, args.attacker_algorithm, device)
    defender_algo = None
    defender_model = None
    if defender_mode == "model":
        defender_algo, defender_model = _load_sb3_model(defender_model_path, args.defender_algorithm, device)

    max_nodes, max_total_services, service_action_limit, firewall_ports, _coverage = _build_defender_bounds(
        envs_folder=envs_folder,
        eval_env_ids=eval_env_ids,
        train_env_ids_for_ports=train_env_ids,
        defender_cfg=defender_cfg,
        defender_model=defender_model,
        logger=logger,
    )

    defender_first_cfg = _as_bool(defender_cfg.get("defender_act_first", False), False)
    defender_first_arg = str(args.defender_first).lower()
    if defender_first_arg == "auto":
        defender_first = defender_first_cfg
    elif defender_first_arg == "true":
        defender_first = True
    elif defender_first_arg == "false":
        defender_first = False
    else:
        raise ValueError("--defender_first must be one of: auto, true, false")

    episode_iterations = int(train_cfg.get("episode_iterations", 200))
    attacker_max_timesteps = int(args.max_steps_per_episode) if int(args.max_steps_per_episode) > 0 else episode_iterations
    defender_default_max = int(defender_cfg.get("defender_max_timesteps", episode_iterations))
    defender_max_timesteps = int(args.max_steps_per_episode) if int(args.max_steps_per_episode) > 0 else defender_default_max

    random_switch_env = RandomSwitchEnv(
        envs_ids=list(eval_env_ids),
        switch_interval=int(args.switch_interval),
        envs_folder=envs_folder,
        save_to_csv=False,
        save_embeddings=False,
        verbose=int(args.verbose),
        training_non_terminal_mode=False,
        training_disable_terminal_rewards=False,
    )
    shared_env = SharedResetEnv(
        random_switch_env,
        proportional_cutoff_coefficient=train_cfg.get("proportional_cutoff_coefficient"),
    )

    event_source = EnvironmentEventSource()
    attacker_env = AttackerEnvWrapper(
        shared_env,
        event_source=event_source,
        max_timesteps=attacker_max_timesteps,
        training_non_terminal_mode=False,
        deadlock_patience=0,
        deadlock_reset_on_no_owned_running=False,
        log_episode_end=False,
        log_episode_summary=bool(attacker_log_episode_summary),
        episode_log_prefix="[eval]",
    )
    defender_env = DefenderEnvWrapper(
        shared_env,
        attacker_reward_store=attacker_env,
        event_source=event_source,
        max_nodes=max_nodes,
        max_total_services=max_total_services,
        service_action_limit=service_action_limit,
        firewall_rule_list=firewall_ports,
        max_timesteps=defender_max_timesteps,
        invalid_action_reward=defender_cfg.get("defender_invalid_action_reward", -50.0),
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
        reimage_auto_focus_detected_node=defender_cfg.get("defender_reimage_auto_focus_detected_node", False),
        blind_firewall_budget=defender_cfg.get("defender_blind_firewall_budget"),
        prioritize_reimage_over_service_actions=defender_cfg.get(
            "defender_prioritize_reimage_over_service_actions", True
        ),
        log_episode_end=False,
        log_episode_summary=bool(defender_log_episode_summary),
        episode_log_prefix="[eval]",
    )

    deterministic = not _as_bool(args.stochastic, False)
    episodes = int(args.episodes)
    safety_max_steps = int(args.safety_max_steps)

    logger.info("Evaluation run folder: %s", run_folder)
    logger.info(
        "Evaluation logs -> eval_log=%s sync_to_run_app_log=%s",
        resolved_eval_log_path,
        _as_bool(args.sync_to_run_app_log, True),
    )
    logger.info("Attacker model: %s (%s)", attacker_model_path, attacker_algo)
    if defender_mode == "model":
        logger.info("Defender model: %s (%s)", defender_model_path, defender_algo)
    logger.info(
        "Defender mode=%s deterministic=%s episodes=%d env_ids=%s defender_first=%s",
        defender_mode,
        deterministic,
        episodes,
        eval_env_ids,
        defender_first,
    )
    logger.info(
        "Episode summary flags -> attacker=%s defender=%s",
        bool(attacker_log_episode_summary),
        bool(defender_log_episode_summary),
    )

    records: List[Dict[str, Any]] = []
    winner_counter: Counter = Counter()
    attacker_reason_counter: Counter = Counter()
    defender_reason_counter: Counter = Counter()

    try:
        for episode_idx in range(1, episodes + 1):
            ep_seed = int(args.seed) + episode_idx - 1
            current_env_id = getattr(random_switch_env, "current_env_index", None)

            attacker_obs, _ = attacker_env.reset(seed=ep_seed)
            defender_obs, _ = defender_env.reset(seed=ep_seed)

            step_pairs = 0
            done = False
            forced_safety_stop = False

            while not done:
                step_pairs += 1

                if defender_first:
                    defender_action = _choose_defender_action(
                        defender_mode,
                        defender_model,
                        defender_obs,
                        defender_env,
                        deterministic,
                    )
                    defender_obs, _dr, d_term, d_trunc, _d_info = defender_env.step(defender_action)
                    attacker_action, _ = attacker_model.predict(attacker_obs, deterministic=deterministic)
                    attacker_obs, _ar, a_term, a_trunc, _a_info = attacker_env.step(_normalize_model_action(attacker_action))
                else:
                    attacker_action, _ = attacker_model.predict(attacker_obs, deterministic=deterministic)
                    attacker_obs, _ar, a_term, a_trunc, _a_info = attacker_env.step(_normalize_model_action(attacker_action))
                    defender_action = _choose_defender_action(
                        defender_mode,
                        defender_model,
                        defender_obs,
                        defender_env,
                        deterministic,
                    )
                    defender_obs, _dr, d_term, d_trunc, _d_info = defender_env.step(defender_action)

                done = bool(a_term or a_trunc or d_term or d_trunc)

                if safety_max_steps > 0 and step_pairs >= safety_max_steps and not done:
                    forced_safety_stop = True
                    logger.warning(
                        "Episode %d reached safety_max_steps=%d, forcing reset.",
                        episode_idx,
                        safety_max_steps,
                    )
                    break

            attacker_reason = _get_wrapper_reason(attacker_env)
            defender_reason = _get_wrapper_reason(defender_env)
            if forced_safety_stop:
                attacker_reason = "safety_max_steps"
                defender_reason = "safety_max_steps"

            attacker_ep_reward = attacker_env.last_episode_reward
            defender_ep_reward = defender_env.last_episode_reward
            if attacker_ep_reward is None:
                attacker_ep_reward = float(np.sum(attacker_env.episode_rewards)) if attacker_env.episode_rewards else 0.0
            if defender_ep_reward is None:
                defender_ep_reward = float(np.sum(defender_env.episode_rewards)) if defender_env.episode_rewards else 0.0

            attacker_ep_len = attacker_env.last_episode_len or step_pairs
            defender_ep_len = defender_env.last_episode_len or step_pairs

            winner = _decide_winner(attacker_reason, defender_reason)
            winner_counter[winner] += 1
            attacker_reason_counter[attacker_reason] += 1
            defender_reason_counter[defender_reason] += 1

            record = {
                "episode": episode_idx,
                "seed": ep_seed,
                "env_id": int(current_env_id) if current_env_id is not None else None,
                "step_pairs": int(step_pairs),
                "attacker_reward": float(attacker_ep_reward),
                "defender_reward": float(defender_ep_reward),
                "attacker_len": int(attacker_ep_len),
                "defender_len": int(defender_ep_len),
                "attacker_reason": attacker_reason,
                "defender_reason": defender_reason,
                "winner": winner,
                "attacker_invalid_actions": int(attacker_env.last_invalid_action_count or 0),
                "defender_invalid_actions": int(defender_env.last_invalid_action_count or 0),
                "defender_sync_skip": int(defender_env.last_episode_sync_skip_count or 0),
                "attacker_owned_ratio": float(attacker_env.last_owned_ratio)
                if attacker_env.last_owned_ratio is not None
                else None,
            }
            records.append(record)

            logger.info(
                "Episode %d | env=%s | steps=%d | winner=%s | A(reward=%.2f,reason=%s) | D(reward=%.2f,reason=%s,invalid=%d)",
                episode_idx,
                str(record["env_id"]),
                step_pairs,
                winner,
                record["attacker_reward"],
                attacker_reason,
                record["defender_reward"],
                defender_reason,
                record["defender_invalid_actions"],
            )

        attacker_rewards = [r["attacker_reward"] for r in records]
        defender_rewards = [r["defender_reward"] for r in records]
        step_counts = [r["step_pairs"] for r in records]
        defender_invalid = [r["defender_invalid_actions"] for r in records]

        summary = {
            "episodes": len(records),
            "attacker_avg_reward": float(np.mean(attacker_rewards)) if attacker_rewards else 0.0,
            "defender_avg_reward": float(np.mean(defender_rewards)) if defender_rewards else 0.0,
            "avg_step_pairs": float(np.mean(step_counts)) if step_counts else 0.0,
            "attacker_win_rate": float(winner_counter.get("attacker", 0) / len(records)) if records else 0.0,
            "defender_win_rate": float(winner_counter.get("defender", 0) / len(records)) if records else 0.0,
            "other_rate": float(winner_counter.get("other", 0) / len(records)) if records else 0.0,
            "avg_defender_invalid_actions": float(np.mean(defender_invalid)) if defender_invalid else 0.0,
            "winner_counts": dict(winner_counter),
            "attacker_reason_counts": dict(attacker_reason_counter),
            "defender_reason_counts": dict(defender_reason_counter),
        }

        result = {
            "timestamp": datetime.now().isoformat(),
            "run_folder": run_folder,
            "envs_folder": envs_folder,
            "env_ids": list(eval_env_ids),
            "defender_mode": defender_mode,
            "deterministic": deterministic,
            "defender_first": bool(defender_first),
            "attacker": {
                "algorithm": attacker_algo,
                "model_path": attacker_model_path,
                "max_timesteps": attacker_max_timesteps,
            },
            "defender": {
                "algorithm": defender_algo if defender_mode == "model" else defender_mode,
                "model_path": defender_model_path,
                "max_timesteps": defender_max_timesteps,
                "max_nodes": max_nodes,
                "service_action_limit": service_action_limit,
                "firewall_ports": firewall_ports,
            },
            "summary": summary,
            "episodes": records,
        }

        output_json = args.output_json
        if output_json:
            if not os.path.isabs(output_json):
                output_json = os.path.abspath(os.path.join(run_folder, output_json))
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info("Saved evaluation report to %s", output_json)

        logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
        return result
    finally:
        try:
            defender_env.close()
        except Exception:
            pass
        try:
            attacker_env.close()
        except Exception:
            pass
        try:
            shared_env.close()
        except Exception:
            pass
        try:
            random_switch_env.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained attacker/defender MARL models with defender mode: model/noop/random"
    )
    parser.add_argument("--run_folder", type=str, required=True, help="MARL run folder containing train_config.yaml and envs/")
    parser.add_argument("--envs_folder", type=str, default=None, help="Optional override for folder containing serialized env *.pkl")
    parser.add_argument("--env_split", choices=["train", "val", "all"], default="val", help="Which split from split.yaml to evaluate")
    parser.add_argument("--env_ids", type=str, default=None, help="Explicit env ids, comma/space separated, overrides --env_split")

    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--switch_interval", type=int, default=1, help="Env switch interval in RandomSwitchEnv")
    parser.add_argument("--max_steps_per_episode", type=int, default=0, help="Override wrapper timeout; 0 means use config defaults")
    parser.add_argument("--safety_max_steps", type=int, default=5000, help="Hard stop guard if episode does not end")

    parser.add_argument("--attacker_model", type=str, default=None, help="Path to attacker model zip; default attacker_model_run_<run_id>.zip")
    parser.add_argument("--attacker_algorithm", type=str, default=None, help="Optional attacker algorithm override")

    parser.add_argument("--defender_mode", choices=["model", "noop", "random"], default="model")
    parser.add_argument("--defender_model", type=str, default=None, help="Path to defender model zip when defender_mode=model")
    parser.add_argument("--defender_algorithm", type=str, default=None, help="Optional defender algorithm override")
    parser.add_argument("--defender_config", type=str, default=None, help="Optional defender config yaml path")

    parser.add_argument("--run_id", type=int, default=1, help="Default model suffix when model path omitted")
    parser.add_argument("--defender_first", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions (default deterministic)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output report path (.json)")
    parser.add_argument("--eval_log_path", type=str, default=None,
                        help="Evaluation app log path (default: <run_folder>/evaluation/app.log)")
    parser.add_argument("--sync_to_run_app_log", choices=["true", "false"], default="true",
                        help="Also append evaluation logs to <run_folder>/app.log")
    parser.add_argument("--attacker_log_episode_summary", choices=["auto", "true", "false"], default="auto",
                        help="Attacker episode summary logging (auto follows train_config.yaml)")
    parser.add_argument("--defender_log_episode_summary", choices=["auto", "true", "false"], default="auto",
                        help="Defender episode summary logging (auto follows defender_config.yaml)")
    parser.add_argument("--verbose", type=int, default=2)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
