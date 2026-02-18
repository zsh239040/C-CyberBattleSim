# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    train_agent_multi_env.py
    Multi-environment training script for C-CyberBattleSim.
"""

import argparse
import copy
import sys
import os
import yaml
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import random
import pickle
from tqdm import tqdm
import shutil
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, HumanOutputFormat, CSVOutputFormat, TensorBoardOutputFormat
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
cyberbattle_root = os.path.join(project_root, "cyberbattle")
sys.path.insert(0, project_root)

from cyberbattle.utils.train_utils import replace_with_classes, check_args, clean_config_save, algorithm_models, reccurrent_algorithms  # noqa: E402
from cyberbattle.utils.math_utils import linear_schedule, calculate_auc, set_seeds  # noqa: E402
from cyberbattle.utils.file_utils import extract_metric_data, load_yaml, save_yaml  # noqa: E402
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv  # noqa: E402
from cyberbattle.agents.multi_env.callbacks_multi_env import EpisodeStatsWrapper, MultiEnvTrainingCallback, MultiEnvValidationCallback  # noqa: E402
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs, wrap_graphs_to_global_envs, wrap_graphs_to_local_envs  # noqa: E402
from cyberbattle._env.static_defender import ScanAndReimageCompromisedMachines, ExternalRandomEvents  # noqa: E402
from cyberbattle.gae.model import GAEEncoder  # noqa: E402
from cyberbattle.utils.log_utils import setup_logging  # noqa: E402

torch.set_default_dtype(torch.float32)
script_dir = os.path.dirname(__file__)

# Resolve the TensorBoard run directory to match SB3 logging output.
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


def _make_env(rank, env_ids, envs_folder, csv_folder, config, seed, switch_interval):
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
        )
        if config["save_csv_file"]:
            env_csv_folder = os.path.join(csv_folder, f"env_{rank}")
            env.update_csv_folder(
                env_csv_folder,
                filename=f"logs_env_{rank}.csv",
                save_embeddings=config["save_embeddings_csv_file"],
                save_to_csv_interval=config["save_csv_file"],
            )
        env = Monitor(env)
        env = EpisodeStatsWrapper(env)
        return env

    return _init


def _build_vec_envs(logs_folder, envs_folder, config, train_ids, val_ids, seed, logger=None):
    if not train_ids:
        raise ValueError("No training environments available.")

    requested_parallel = config["num_parallel_envs"]
    num_parallel_envs = min(requested_parallel, len(train_ids))
    if requested_parallel > len(train_ids) and logger:
        logger.info(
            "Requested %d parallel envs, but only %d training envs are available. Capping to %d.",
            requested_parallel, len(train_ids), num_parallel_envs
        )
    config["num_parallel_envs"] = num_parallel_envs

    env_id_groups = _split_env_ids(
        train_ids,
        num_parallel_envs,
        config["env_sampling_mode"],
        seed,
        config.get("shuffle_train_envs", True),
    )
    csv_folder = os.path.join(logs_folder, "csv")
    vec_env_fns = [
        _make_env(
            rank,
            env_id_groups[rank],
            envs_folder,
            csv_folder,
            config,
            seed,
            config["switch_interval"],
        )
        for rank in range(num_parallel_envs)
    ]
    if config["vec_env_type"] == "subproc":
        train_envs = SubprocVecEnv(vec_env_fns, start_method=config["subproc_start_method"])
    else:
        train_envs = DummyVecEnv(vec_env_fns)

    train_envs = VecNormalize(train_envs, norm_obs=config["norm_obs"], norm_reward=config["norm_reward"])

    val_envs = None
    if val_ids:
        val_csv_folder = os.path.join(logs_folder, "validation", "csv")
        val_env_fn = _make_env(
            0,
            val_ids,
            envs_folder,
            val_csv_folder,
            config,
            seed,
            config["val_switch_interval"],
        )
        val_envs = DummyVecEnv([val_env_fn])
        val_envs = VecNormalize(val_envs, norm_obs=config["norm_obs"], norm_reward=False)

    return train_envs, val_envs


def train_rl_algorithm(logs_folder, envs_folder, config, train_ids, val_ids, metric_name=None, logger=None, verbose=1):
    metric_values = []
    resume_enabled = bool(config.get("resume_from"))
    if verbose:
        logger.info(
            "Training algorithm %s on %d environments with %d runs.",
            config["algorithm"], config["num_environments"], config["num_runs"]
        )
    for run_id in range(config["num_runs"]):
        seed = config["seeds_runs"][run_id]
        set_seeds(seed)
        if verbose:
            logger.info("Run %d/%d with seed %d started.", run_id + 1, config["num_runs"], seed)

        completion_flag = _run_completion_flag_path(logs_folder, run_id + 1)
        if resume_enabled and os.path.exists(completion_flag):
            if verbose:
                logger.info("Run %d already completed, skipping training.", run_id + 1)
        else:
            train_envs, val_envs = _build_vec_envs(
                logs_folder, envs_folder, config, train_ids, val_ids, seed, logger=logger
            )

            if verbose >= 2:
                sample_env = _make_env(
                    0,
                    [train_ids[0]],
                    envs_folder,
                    os.path.join(logs_folder, "csv"),
                    config,
                    seed,
                    config["switch_interval"],
                )()
                check_env(sample_env)
                sample_env.close()

            train_model(
                train_envs,
                logs_folder,
                config,
                run_id + 1,
                val_envs=val_envs,
                logger=logger,
                verbose=verbose,
                resume=resume_enabled,
            )

        if metric_name:
            tensorboard_dir = _get_tensorboard_dir(logs_folder, config["algorithm"], run_id + 1)
            times, values = extract_metric_data(tensorboard_dir, metric_name)
            auc = calculate_auc(times, values)
            if verbose:
                logger.info("The AUC of metric %s for the run %d is %s", metric_name, run_id + 1, auc)
            metric_values.append(auc)

    if verbose:
        logger.info("Training finished.")
    return metric_values


def train_model(train_envs, logs_folder, config, run_id, val_envs=None, logger=None, verbose=1, resume=False):
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info("Training on device: %s", device)

    algorithm_config = copy.deepcopy(config["algorithm_hyperparams"])
    if algorithm_config["learning_rate_type"] == "linear":
        learning_rate = linear_schedule(algorithm_config["learning_rate"], algorithm_config["learning_rate_final"])
    else:
        learning_rate = algorithm_config["learning_rate"]

    algorithm_config.pop("learning_rate_type", None)
    algorithm_config.pop("learning_rate", None)
    algorithm_config.pop("learning_rate_final", None)

    model_class = algorithm_models[config["algorithm"]]

    if verbose:
        logger.info("Normalization of observations: %s", config["norm_obs"])
        logger.info("Normalization of rewards: %s", config["norm_reward"])

    if "batch_size" in algorithm_config:
        total_batch = algorithm_config.get("n_steps", 1) * train_envs.num_envs
        if algorithm_config["batch_size"] > total_batch and logger:
            logger.warning(
                "batch_size (%d) is larger than n_steps * n_envs (%d). Consider reducing batch_size.",
                algorithm_config["batch_size"], total_batch
            )

    resume_from = None
    if resume:
        checkpoint_dir = os.path.join(logs_folder, "checkpoints", str(run_id))
        resume_from = _find_latest_checkpoint(checkpoint_dir)
        if resume_from and verbose:
            logger.info("Resuming from checkpoint %s", resume_from)

    if resume_from:
        model = model_class.load(resume_from, env=train_envs, device=device)
    elif config["algorithm"] in reccurrent_algorithms:
        if config["finetune_model"] and os.path.exists(config["finetune_model"]):
            model = model_class.load(config["finetune_model"], env=train_envs, device=device)
            if verbose:
                logger.info("Loaded model to finetune from %s", config["finetune_model"])
        else:
            if verbose:
                logger.info("Initialized new model from scratch")
            model = model_class(
                "MultiInputLstmPolicy",
                train_envs,
                policy_kwargs=config["policy_kwargs"],
                learning_rate=learning_rate,
                tensorboard_log=logs_folder,
                **algorithm_config,
                verbose=verbose,
                device=device,
            )
    else:
        lstm_keys = ["lstm_hidden_size", "n_lstm_layers"]
        for key in lstm_keys:
            if key in config["policy_kwargs"]:
                del config["policy_kwargs"][key]
        if config.get("finetune_model") and os.path.exists(config["finetune_model"]):
            model = model_class.load(
                config["finetune_model"],
                tensorboard_log=logs_folder,
                env=train_envs,
                verbose=1,
                learning_rate=learning_rate,
                device=device,
            )
            if verbose:
                logger.info("Loaded model to finetune from %s", config["finetune_model"])
        else:
            if verbose:
                logger.info("Initialized new model from scratch")
            model = model_class(
                "MultiInputPolicy",
                train_envs,
                policy_kwargs=config["policy_kwargs"],
                learning_rate=learning_rate,
                tensorboard_log=logs_folder,
                **algorithm_config,
                verbose=verbose,
                device=device,
            )

    app_log_path = os.path.join(logs_folder, "app.log")
    tensorboard_dir = _get_tensorboard_dir(logs_folder, config["algorithm"], run_id)
    os.makedirs(tensorboard_dir, exist_ok=True)
    output_formats = [
        HumanOutputFormat(sys.stdout),
        HumanOutputFormat(open(app_log_path, "a")),
        CSVOutputFormat(os.path.join(logs_folder, "progress.csv")),
        TensorBoardOutputFormat(tensorboard_dir),
    ]
    sb3_logger = Logger(logs_folder, output_formats)
    model.set_logger(sb3_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoints_save_freq"],
        save_path=os.path.join(logs_folder, "checkpoints", str(run_id)),
        name_prefix="checkpoint",
    )
    train_callback = MultiEnvTrainingCallback(verbose=verbose)

    callbacks = [checkpoint_callback, train_callback]

    if val_envs:
        val_callback = MultiEnvValidationCallback(
            val_env=val_envs,
            n_val_episodes=config["n_val_episodes"],
            val_freq=config["val_freq"],
            validation_folder_path=os.path.join(logs_folder, "validation", str(run_id)),
            early_stopping=config["early_stopping"],
            patience=config["early_stopping"],
            output_logger=logger,
            verbose=verbose,
        )
        callbacks.append(val_callback)
    if verbose:
        logger.info(
            "Training started for run %d lasting %d iterations with episode length %d",
            run_id, config["train_iterations"], config["episode_iterations"]
        )
    try:
        model.learn(
            total_timesteps=config["train_iterations"],
            callback=callbacks,
            reset_num_timesteps=not bool(resume_from),
        )
    except Exception as e:
        logger.error("Training failed due to errors: %s, skipping this run", e)
        return
    with open(_run_completion_flag_path(logs_folder, run_id), "w") as f:
        f.write(f"completed at {datetime.now().isoformat()}\n")


def setup_train_via_args(args, logs_folder=None, envs_folder=None):
    args.yaml = args.yaml if "yaml" in args else False
    if args.yaml:
        with open(os.path.join(script_dir, "logs", str(args.yaml), "train_config.yaml"), "r") as config_file:
            config = yaml.safe_load(config_file)
        args.goal = config["goal"]
        args.nlp_extractor = config["nlp_extractor"]
        args.algorithm = config["algorithm"]
        config.setdefault("save_csv_file", args.save_csv_file)
        config.setdefault("save_embeddings_csv_file", args.save_embeddings_csv_file)
        config.setdefault("verbose", args.verbose)
        config.setdefault("num_parallel_envs", args.num_parallel_envs)
        config.setdefault("env_sampling_mode", args.env_sampling_mode)
        config.setdefault("shuffle_train_envs", args.shuffle_train_envs)
        config.setdefault("max_train_envs", args.max_train_envs)
        config.setdefault("vec_env_type", args.vec_env_type)
        config.setdefault("subproc_start_method", args.subproc_start_method)
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

    if "yaml" in args and args.yaml:
        config["yaml"] = args.yaml
    else:
        if args.verbose:
            logger.info("Reading default configuration YAML files")
        with open(os.path.join(script_dir, args.train_config), "r") as config_file:
            config = yaml.safe_load(config_file)
        with open(os.path.join(script_dir, args.rewards_config), "r") as config_file:
            rewards_config = yaml.safe_load(config_file)
        with open(os.path.join(script_dir, args.algo_config), "r") as config_file:
            algorithm_config = yaml.safe_load(config_file)
        args.algorithm_hyperparams = algorithm_config[args.algorithm]
        args.rewards_dict = rewards_config["rewards_dict"][args.goal]
        args.penalties_dict = rewards_config["penalties_dict"][args.goal]
        args.policy_kwargs = algorithm_config["policy_kwargs"]
        config.update(vars(args))

    if "finetune_model" in args and args.finetune_model:
        if ".zip" not in args.finetune_model:
            raise ValueError("The path provided should be the zip file related to the model to finetune!")
        config["finetune_model"] = os.path.join(script_dir, "logs", config["finetune_model"])

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
        seeds_loaded = load_yaml(os.path.join(script_dir, args.load_seeds, "seeds.yaml"))
        seeds_runs = seeds_loaded["seeds"][0:config["num_runs"]]

    save_yaml({"seeds": seeds_runs}, logs_folder, "seeds.yaml")
    config.update({"seeds_runs": seeds_runs})
    set_seeds(seeds_runs[0])

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    def get_base_path(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        run_dirs = sorted([d for d in subdirs if re.match(r"run_\d+", d)])
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
    parser = argparse.ArgumentParser(description="Train RL algorithm on multiple C-CyberBattleSim environments!")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "a2c", "rppo", "trpo", "ddpg", "sac", "td3", "tqc"], default="trpo")
    parser.add_argument("--environment_type", type=str, choices=["continuous", "global", "local"], default="continuous")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("-nlp", "--nlp_extractor", type=str, choices=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT", "SecRoBERTa"], default="CySecBERT")
    parser.add_argument("--holdout", action="store_true", default=True)
    parser.add_argument("--finetune_model", type=str, help="Path to the model to eventually finetune (relative to the logs folder)")
    parser.add_argument("--early_stopping", type=int, default=7)
    parser.add_argument("-g", "--goal", type=str, default="control", choices=["control", "discovery", "disruption", "control_node", "discovery_node", "disruption_node"])
    parser.add_argument("--yaml", default=False, help="Read configuration file from YAML file of a previous training")
    parser.add_argument("--name", default="ALGO_MULTI", help="Name of the logs folder related to the run")
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
    parser.add_argument("--max_train_envs", type=int, default=None, help="Limit number of training environments (n)")
    parser.add_argument("--vec_env_type", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--subproc_start_method", choices=["fork", "spawn", "forkserver"], default="spawn")

    args = parser.parse_args()

    error, message = check_args(args)
    if error:
        raise ValueError(message)

    general_config = load_yaml(os.path.join(project_root, "config.yaml"))
    if not args.load_envs and not args.load_processed_envs:
        args.load_envs = general_config["default_environments_path"]

    logger, logs_folder, envs_folder, config, train_ids, val_ids = setup_train_via_args(args, None)
    train_rl_algorithm(logs_folder, envs_folder, config, train_ids, val_ids, logger=logger, verbose=args.verbose)
