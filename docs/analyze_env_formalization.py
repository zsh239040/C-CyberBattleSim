#!/usr/bin/env python3
"""
Analyze C-CyberBattleSim environment samples for Chapter 2 formalization.

Outputs:
- JSON with full statistics
- Markdown summary with key values that can be cited in paper text
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

# Ensure pickled objects (cyberbattle.* modules) are importable when script runs from docs/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = _safe_mean(values)
    return float(math.sqrt(sum((x - mu) ** 2 for x in values) / len(values)))


def _safe_min(values: Sequence[float]) -> float:
    return float(min(values)) if values else 0.0


def _safe_max(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def _summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": _safe_mean(values),
        "std": _safe_std(values),
        "min": _safe_min(values),
        "max": _safe_max(values),
    }


def _top(counter: Counter, k: int = 15) -> List[Dict[str, object]]:
    return [{"name": name, "count": int(count)} for name, count in counter.most_common(k)]


def _find_model_file(env_dir: Path, preferred_name: str) -> Optional[Path]:
    preferred = env_dir / preferred_name
    if preferred.exists():
        return preferred

    candidates = sorted(
        [
            p
            for p in env_dir.glob("network_*.pkl")
            if p.name != "network_graph.pkl"
        ]
    )
    return candidates[0] if candidates else None


def _read_split(split_path: Path) -> Dict[str, object]:
    import yaml

    if not split_path.exists():
        return {
            "available": False,
            "training_ids": [],
            "validation_ids": [],
            "test_ids": [],
        }

    with split_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    training_ids = [int(item["id"]) for item in data.get("training_set", []) if "id" in item]
    validation_ids = [int(item["id"]) for item in data.get("validation_set", []) if "id" in item]
    test_ids = [int(item["id"]) for item in data.get("test_set", []) if "id" in item]

    return {
        "available": True,
        "params": data.get("params", {}),
        "training_ids": training_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
    }


def _int_status_name(status_obj: object) -> str:
    # status objects in model are enums; keep robust for pickled variants
    if hasattr(status_obj, "name"):
        return str(getattr(status_obj, "name"))
    return str(status_obj)


def _int_priv_name(priv_obj: object) -> str:
    if hasattr(priv_obj, "name"):
        return str(getattr(priv_obj, "name"))
    return str(priv_obj)


def _analyze_single_env(env_id: int, model_file: Path) -> Dict[str, object]:
    with model_file.open("rb") as f:
        model_obj = pickle.load(f)

    network = model_obj.network
    knows_graph = model_obj.knows_graph
    access_graph = model_obj.access_graph
    dos_graph = model_obj.dos_graph

    nodes = list(network.nodes)
    n_nodes = len(nodes)
    directed_possible = max(1, n_nodes * (n_nodes - 1))

    e_knows = int(knows_graph.number_of_edges())
    e_access = int(access_graph.number_of_edges())
    e_dos = int(dos_graph.number_of_edges())

    edge_set_knows = set(knows_graph.edges())
    edge_set_access = set(access_graph.edges())
    edge_set_dos = set(dos_graph.edges())

    node_tag_counts: Counter = Counter()
    node_status_counts: Counter = Counter()
    level_at_access_counts: Counter = Counter()
    privilege_level_counts: Counter = Counter()

    services_per_node: List[int] = []
    running_services_per_node: List[int] = []
    vulnerabilities_per_node: List[int] = []

    has_data_count = 0
    visible_count = 0
    reimageable_count = 0
    persistence_true_count = 0
    defense_evasion_true_count = 0

    incoming_fw_rule_count = 0
    outgoing_fw_rule_count = 0
    incoming_fw_blocked_count = 0
    outgoing_fw_blocked_count = 0

    service_port_counts: Counter = Counter()

    vuln_instance_count = 0
    vuln_unique_ids: set = set()
    attack_vector_counts: Counter = Counter()
    attack_complexity_counts: Counter = Counter()
    base_severity_counts: Counter = Counter()
    privileges_required_counts: Counter = Counter()
    vuln_type_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    vuln_results_len: List[int] = []

    cvss_base_scores: List[float] = []
    cvss_impact_scores: List[float] = []
    cvss_exploitability_scores: List[float] = []

    success_rates: List[float] = []
    probing_detection_rates: List[float] = []
    exploit_detection_rates: List[float] = []

    access_edge_vuln_refs = 0
    dos_edge_vuln_refs = 0

    for _, _, data in access_graph.edges(data=True):
        vulns = data.get("vulnerabilities", [])
        access_edge_vuln_refs += len(vulns)

    for _, _, data in dos_graph.edges(data=True):
        vulns = data.get("vulnerabilities", [])
        dos_edge_vuln_refs += len(vulns)

    for node_id in nodes:
        node_info = network.nodes[node_id]["data"]

        node_tag_counts[str(getattr(node_info, "tag", ""))] += 1
        node_status_counts[_int_status_name(getattr(node_info, "status", ""))] += 1
        level_at_access_counts[_int_priv_name(getattr(node_info, "level_at_access", ""))] += 1
        privilege_level_counts[_int_priv_name(getattr(node_info, "privilege_level", ""))] += 1

        services = list(getattr(node_info, "services", []))
        services_per_node.append(len(services))
        running_services_per_node.append(sum(1 for s in services if bool(getattr(s, "running", False))))

        for service in services:
            service_port_counts[str(getattr(service, "name", ""))] += 1

        if bool(getattr(node_info, "has_data", False)):
            has_data_count += 1
        if bool(getattr(node_info, "visible", False)):
            visible_count += 1
        if bool(getattr(node_info, "reimageable", False)):
            reimageable_count += 1
        if bool(getattr(node_info, "persistence", False)):
            persistence_true_count += 1
        if bool(getattr(node_info, "defense_evasion", False)):
            defense_evasion_true_count += 1

        fw = getattr(node_info, "firewall", None)
        incoming_rules = list(getattr(fw, "incoming", [])) if fw is not None else []
        outgoing_rules = list(getattr(fw, "outgoing", [])) if fw is not None else []

        incoming_fw_rule_count += len(incoming_rules)
        outgoing_fw_rule_count += len(outgoing_rules)

        for rule in incoming_rules:
            perm = str(getattr(getattr(rule, "permission", None), "name", getattr(rule, "permission", "")))
            if perm.upper() == "BLOCK":
                incoming_fw_blocked_count += 1

        for rule in outgoing_rules:
            perm = str(getattr(getattr(rule, "permission", None), "name", getattr(rule, "permission", "")))
            if perm.upper() == "BLOCK":
                outgoing_fw_blocked_count += 1

        vulns = getattr(node_info, "vulnerabilities", {})
        vulnerabilities_per_node.append(len(vulns))

        for vuln_id, vuln in vulns.items():
            vuln_instance_count += 1
            vuln_unique_ids.add(str(vuln_id))

            attack_vector_counts[str(getattr(vuln, "attack_vector", ""))] += 1
            attack_complexity_counts[str(getattr(vuln, "attack_complexity", ""))] += 1
            base_severity_counts[str(getattr(vuln, "base_severity", ""))] += 1
            privileges_required_counts[_int_priv_name(getattr(vuln, "privileges_required", ""))] += 1

            cvss_base_scores.append(float(getattr(vuln, "base_score", 0.0) or 0.0))
            cvss_impact_scores.append(float(getattr(vuln, "impact_score", 0.0) or 0.0))
            cvss_exploitability_scores.append(float(getattr(vuln, "exploitability_score", 0.0) or 0.0))

            rates = getattr(vuln, "rates", None)
            if rates is not None:
                success_rates.append(float(getattr(rates, "successRate", 0.0) or 0.0))
                probing_detection_rates.append(float(getattr(rates, "probingDetectionRate", 0.0) or 0.0))
                exploit_detection_rates.append(float(getattr(rates, "exploitDetectionRate", 0.0) or 0.0))

            results = list(getattr(vuln, "results", []))
            vuln_results_len.append(len(results))
            for r in results:
                type_str = str(getattr(r, "type_str", ""))
                vuln_type_counts[type_str] += 1
                out = getattr(r, "outcome", None)
                out_name = out.__class__.__name__ if out is not None else "None"
                outcome_counts[out_name] += 1

    return {
        "env_id": env_id,
        "model_file": str(model_file),
        "nodes": n_nodes,
        "edges": {
            "knows": e_knows,
            "access": e_access,
            "dos": e_dos,
        },
        "edge_density": {
            "knows": e_knows / directed_possible,
            "access": e_access / directed_possible,
            "dos": e_dos / directed_possible,
        },
        "edge_overlap": {
            "knows_access": len(edge_set_knows & edge_set_access),
            "knows_dos": len(edge_set_knows & edge_set_dos),
            "access_dos": len(edge_set_access & edge_set_dos),
            "all_three": len(edge_set_knows & edge_set_access & edge_set_dos),
        },
        "node_stats": {
            "tag_counts": dict(node_tag_counts),
            "status_counts": dict(node_status_counts),
            "level_at_access_counts": dict(level_at_access_counts),
            "privilege_level_counts": dict(privilege_level_counts),
            "services_per_node": _summary(services_per_node),
            "running_services_per_node": _summary(running_services_per_node),
            "vulnerabilities_per_node": _summary(vulnerabilities_per_node),
            "has_data_ratio": has_data_count / max(1, n_nodes),
            "visible_ratio": visible_count / max(1, n_nodes),
            "reimageable_ratio": reimageable_count / max(1, n_nodes),
            "persistence_true_ratio": persistence_true_count / max(1, n_nodes),
            "defense_evasion_true_ratio": defense_evasion_true_count / max(1, n_nodes),
        },
        "firewall_stats": {
            "incoming_rule_count": incoming_fw_rule_count,
            "outgoing_rule_count": outgoing_fw_rule_count,
            "incoming_block_ratio": incoming_fw_blocked_count / max(1, incoming_fw_rule_count),
            "outgoing_block_ratio": outgoing_fw_blocked_count / max(1, outgoing_fw_rule_count),
        },
        "service_port_top": _top(service_port_counts, k=12),
        "vulnerability_stats": {
            "vulnerability_instances": vuln_instance_count,
            "unique_vulnerability_ids": len(vuln_unique_ids),
            "attack_vector_counts": dict(attack_vector_counts),
            "attack_complexity_counts": dict(attack_complexity_counts),
            "base_severity_counts": dict(base_severity_counts),
            "privileges_required_counts": dict(privileges_required_counts),
            "predicted_result_type_counts": dict(vuln_type_counts),
            "predicted_outcome_counts": dict(outcome_counts),
            "results_per_vulnerability": _summary(vuln_results_len),
            "base_score": _summary(cvss_base_scores),
            "impact_score": _summary(cvss_impact_scores),
            "exploitability_score": _summary(cvss_exploitability_scores),
            "success_rate": _summary(success_rates),
            "probing_detection_rate": _summary(probing_detection_rates),
            "exploit_detection_rate": _summary(exploit_detection_rates),
        },
        "edge_vulnerability_refs": {
            "access": access_edge_vuln_refs,
            "dos": dos_edge_vuln_refs,
        },
    }


def _aggregate_env_stats(per_env: List[Dict[str, object]], split_info: Dict[str, object]) -> Dict[str, object]:
    if not per_env:
        return {}

    by_id = {int(item["env_id"]): item for item in per_env}

    nodes_values = [float(item["nodes"]) for item in per_env]
    e_knows_values = [float(item["edges"]["knows"]) for item in per_env]
    e_access_values = [float(item["edges"]["access"]) for item in per_env]
    e_dos_values = [float(item["edges"]["dos"]) for item in per_env]

    d_knows_values = [float(item["edge_density"]["knows"]) for item in per_env]
    d_access_values = [float(item["edge_density"]["access"]) for item in per_env]
    d_dos_values = [float(item["edge_density"]["dos"]) for item in per_env]

    has_data_ratios = [float(item["node_stats"]["has_data_ratio"]) for item in per_env]
    visible_ratios = [float(item["node_stats"]["visible_ratio"]) for item in per_env]
    vuln_per_node_means = [float(item["node_stats"]["vulnerabilities_per_node"]["mean"]) for item in per_env]
    service_per_node_means = [float(item["node_stats"]["services_per_node"]["mean"]) for item in per_env]

    incoming_block_ratios = [float(item["firewall_stats"]["incoming_block_ratio"]) for item in per_env]
    outgoing_block_ratios = [float(item["firewall_stats"]["outgoing_block_ratio"]) for item in per_env]

    vuln_instances = [float(item["vulnerability_stats"]["vulnerability_instances"]) for item in per_env]
    unique_vuln_ids = [float(item["vulnerability_stats"]["unique_vulnerability_ids"]) for item in per_env]

    global_outcome_counts: Counter = Counter()
    global_attack_vector_counts: Counter = Counter()
    global_attack_complexity_counts: Counter = Counter()
    global_severity_counts: Counter = Counter()
    global_priv_required_counts: Counter = Counter()

    port_counter: Counter = Counter()
    node_tag_counter: Counter = Counter()
    status_counter: Counter = Counter()
    level_at_access_counter: Counter = Counter()

    base_score_values: List[float] = []
    impact_score_values: List[float] = []
    exploitability_values: List[float] = []
    success_rate_values: List[float] = []

    access_edge_vuln_refs_values: List[float] = []
    dos_edge_vuln_refs_values: List[float] = []

    for item in per_env:
        for k, v in item["vulnerability_stats"]["predicted_outcome_counts"].items():
            global_outcome_counts[k] += int(v)
        for k, v in item["vulnerability_stats"]["attack_vector_counts"].items():
            global_attack_vector_counts[k] += int(v)
        for k, v in item["vulnerability_stats"]["attack_complexity_counts"].items():
            global_attack_complexity_counts[k] += int(v)
        for k, v in item["vulnerability_stats"]["base_severity_counts"].items():
            global_severity_counts[k] += int(v)
        for k, v in item["vulnerability_stats"]["privileges_required_counts"].items():
            global_priv_required_counts[k] += int(v)

        for p in item.get("service_port_top", []):
            port_counter[p["name"]] += int(p["count"])

        for k, v in item["node_stats"]["tag_counts"].items():
            node_tag_counter[k] += int(v)
        for k, v in item["node_stats"]["status_counts"].items():
            status_counter[k] += int(v)
        for k, v in item["node_stats"]["level_at_access_counts"].items():
            level_at_access_counter[k] += int(v)

        base_score_values.append(float(item["vulnerability_stats"]["base_score"]["mean"]))
        impact_score_values.append(float(item["vulnerability_stats"]["impact_score"]["mean"]))
        exploitability_values.append(float(item["vulnerability_stats"]["exploitability_score"]["mean"]))
        success_rate_values.append(float(item["vulnerability_stats"]["success_rate"]["mean"]))

        access_edge_vuln_refs_values.append(float(item["edge_vulnerability_refs"]["access"]))
        dos_edge_vuln_refs_values.append(float(item["edge_vulnerability_refs"]["dos"]))

    def _subset_summary(env_ids: Iterable[int]) -> Dict[str, object]:
        selected = [by_id[eid] for eid in env_ids if eid in by_id]
        if not selected:
            return {"count": 0}
        return {
            "count": len(selected),
            "nodes_mean": _safe_mean([float(s["nodes"]) for s in selected]),
            "knows_edges_mean": _safe_mean([float(s["edges"]["knows"]) for s in selected]),
            "access_edges_mean": _safe_mean([float(s["edges"]["access"]) for s in selected]),
            "dos_edges_mean": _safe_mean([float(s["edges"]["dos"]) for s in selected]),
            "vuln_instances_mean": _safe_mean([float(s["vulnerability_stats"]["vulnerability_instances"]) for s in selected]),
        }

    training_ids = split_info.get("training_ids", []) if isinstance(split_info, dict) else []
    validation_ids = split_info.get("validation_ids", []) if isinstance(split_info, dict) else []
    test_ids = split_info.get("test_ids", []) if isinstance(split_info, dict) else []

    return {
        "num_envs": len(per_env),
        "nodes": _summary(nodes_values),
        "edges": {
            "knows": _summary(e_knows_values),
            "access": _summary(e_access_values),
            "dos": _summary(e_dos_values),
        },
        "edge_density": {
            "knows": _summary(d_knows_values),
            "access": _summary(d_access_values),
            "dos": _summary(d_dos_values),
        },
        "node_ratios": {
            "has_data": _summary(has_data_ratios),
            "visible": _summary(visible_ratios),
        },
        "node_shape": {
            "services_per_node_mean": _summary(service_per_node_means),
            "vulnerabilities_per_node_mean": _summary(vuln_per_node_means),
        },
        "firewall": {
            "incoming_block_ratio": _summary(incoming_block_ratios),
            "outgoing_block_ratio": _summary(outgoing_block_ratios),
        },
        "vulnerability_volume": {
            "instances": _summary(vuln_instances),
            "unique_ids": _summary(unique_vuln_ids),
        },
        "vulnerability_profile": {
            "attack_vector_counts": dict(global_attack_vector_counts),
            "attack_complexity_counts": dict(global_attack_complexity_counts),
            "base_severity_counts": dict(global_severity_counts),
            "privileges_required_counts": dict(global_priv_required_counts),
            "predicted_outcome_top": _top(global_outcome_counts, k=20),
            "base_score_means": _summary(base_score_values),
            "impact_score_means": _summary(impact_score_values),
            "exploitability_score_means": _summary(exploitability_values),
            "success_rate_means": _summary(success_rate_values),
        },
        "edge_vulnerability_refs": {
            "access": _summary(access_edge_vuln_refs_values),
            "dos": _summary(dos_edge_vuln_refs_values),
        },
        "top_ports": _top(port_counter, k=20),
        "node_tags": dict(node_tag_counter),
        "status_counts": dict(status_counter),
        "level_at_access_counts": dict(level_at_access_counter),
        "split_consistency": {
            "training": _subset_summary(training_ids),
            "validation": _subset_summary(validation_ids),
            "test": _subset_summary(test_ids),
        },
    }


def _render_markdown(report: Dict[str, object]) -> str:
    split_info = report.get("split_info", {})
    agg = report.get("aggregate", {})

    def _fmt(v: float, digits: int = 4) -> str:
        return f"{float(v):.{digits}f}"

    lines: List[str] = []
    lines.append("# C-CyberBattleSim Chapter-2 Environment Statistics")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Dataset root: `{report.get('dataset_root', '')}`")
    lines.append(f"- Model file preference: `{report.get('model_file_preference', '')}`")
    lines.append(f"- Number of analyzed environments: {agg.get('num_envs', 0)}")

    if isinstance(split_info, dict) and split_info.get("available"):
        lines.append("- Split (from split.yaml):")
        lines.append(f"  - train: {len(split_info.get('training_ids', []))} envs")
        lines.append(f"  - validation: {len(split_info.get('validation_ids', []))} envs")
        lines.append(f"  - test: {len(split_info.get('test_ids', []))} envs")

    lines.append("")
    lines.append("## Topology-Level")
    nodes = agg.get("nodes", {})
    edges = agg.get("edges", {})
    density = agg.get("edge_density", {})
    lines.append(
        f"- Nodes per env: mean={_fmt(nodes.get('mean', 0), 2)}, std={_fmt(nodes.get('std', 0), 2)}, "
        f"min={int(nodes.get('min', 0))}, max={int(nodes.get('max', 0))}"
    )
    for edge_key in ("knows", "access", "dos"):
        edge_item = edges.get(edge_key, {})
        den_item = density.get(edge_key, {})
        lines.append(
            f"- {edge_key} edges: mean={_fmt(edge_item.get('mean', 0), 2)}, std={_fmt(edge_item.get('std', 0), 2)}, "
            f"density mean={_fmt(den_item.get('mean', 0), 4)}"
        )

    lines.append("")
    lines.append("## Node and Firewall")
    node_ratios = agg.get("node_ratios", {})
    node_shape = agg.get("node_shape", {})
    fw = agg.get("firewall", {})
    lines.append(f"- has_data ratio mean: {_fmt(node_ratios.get('has_data', {}).get('mean', 0), 4)}")
    lines.append(f"- visible ratio mean: {_fmt(node_ratios.get('visible', {}).get('mean', 0), 4)}")
    lines.append(
        f"- services per node (env means): mean={_fmt(node_shape.get('services_per_node_mean', {}).get('mean', 0), 2)}"
    )
    lines.append(
        f"- vulnerabilities per node (env means): mean={_fmt(node_shape.get('vulnerabilities_per_node_mean', {}).get('mean', 0), 2)}"
    )
    lines.append(f"- incoming firewall BLOCK ratio mean: {_fmt(fw.get('incoming_block_ratio', {}).get('mean', 0), 4)}")
    lines.append(f"- outgoing firewall BLOCK ratio mean: {_fmt(fw.get('outgoing_block_ratio', {}).get('mean', 0), 4)}")

    lines.append("")
    lines.append("## Vulnerability Profile")
    vp = agg.get("vulnerability_profile", {})
    vv = agg.get("vulnerability_volume", {})
    lines.append(
        f"- vulnerability instances per env: mean={_fmt(vv.get('instances', {}).get('mean', 0), 2)}"
    )
    lines.append(
        f"- unique vulnerability IDs per env: mean={_fmt(vv.get('unique_ids', {}).get('mean', 0), 2)}"
    )
    lines.append(f"- attack vectors: {vp.get('attack_vector_counts', {})}")
    lines.append(f"- attack complexity: {vp.get('attack_complexity_counts', {})}")
    lines.append(f"- base severity: {vp.get('base_severity_counts', {})}")
    lines.append(f"- privileges required: {vp.get('privileges_required_counts', {})}")
    lines.append(
        f"- base score (env means): mean={_fmt(vp.get('base_score_means', {}).get('mean', 0), 3)}"
    )
    lines.append(
        f"- exploitability score (env means): mean={_fmt(vp.get('exploitability_score_means', {}).get('mean', 0), 3)}"
    )
    lines.append(
        f"- impact score (env means): mean={_fmt(vp.get('impact_score_means', {}).get('mean', 0), 3)}"
    )
    lines.append(
        f"- success rate (env means): mean={_fmt(vp.get('success_rate_means', {}).get('mean', 0), 4)}"
    )

    lines.append("")
    lines.append("## Predicted Outcomes (Top 12)")
    for row in vp.get("predicted_outcome_top", [])[:12]:
        lines.append(f"- {row['name']}: {row['count']}")

    lines.append("")
    lines.append("## Split Consistency")
    split_cons = agg.get("split_consistency", {})
    for split_name in ("training", "validation", "test"):
        s = split_cons.get(split_name, {})
        lines.append(
            f"- {split_name}: count={s.get('count', 0)}, nodes_mean={_fmt(s.get('nodes_mean', 0), 2)}, "
            f"knows/access/dos edges mean={_fmt(s.get('knows_edges_mean', 0), 2)}/"
            f"{_fmt(s.get('access_edges_mean', 0), 2)}/{_fmt(s.get('dos_edges_mean', 0), 2)}"
        )

    lines.append("")
    lines.append("## Top Ports (frequency)")
    for row in agg.get("top_ports", [])[:15]:
        lines.append(f"- {row['name']}: {row['count']}")

    return "\n".join(lines) + "\n"


def analyze(dataset_root: Path, preferred_model_file: str) -> Dict[str, object]:
    split_info = _read_split(dataset_root / "split.yaml")

    env_dirs = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )

    per_env: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for env_dir in env_dirs:
        env_id = int(env_dir.name)
        model_file = _find_model_file(env_dir, preferred_model_file)
        if model_file is None:
            skipped.append({"env_id": env_id, "reason": "no_model_file"})
            continue
        try:
            per_env.append(_analyze_single_env(env_id, model_file))
        except Exception as exc:  # keep robust on partial corruption
            skipped.append({"env_id": env_id, "reason": f"load_error: {exc}"})

    aggregate = _aggregate_env_stats(per_env, split_info)

    return {
        "dataset_root": str(dataset_root),
        "model_file_preference": preferred_model_file,
        "split_info": split_info,
        "aggregate": aggregate,
        "per_env": per_env,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze C-CyberBattleSim env samples for chapter formalization")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cyberbattle/data/env_samples/syntethic_deployment_20_graphs_100_nodes"),
        help="Path to environment sample root",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="network_CySecBERT.pkl",
        help="Preferred per-env model filename",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/ch2_env_stats.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/ch2_env_stats.md"),
        help="Output Markdown summary path",
    )

    args = parser.parse_args()

    report = analyze(args.dataset_root, args.model_file)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved Markdown: {args.output_md}")
    print(f"Analyzed environments: {report.get('aggregate', {}).get('num_envs', 0)}")
    print(f"Skipped environments: {len(report.get('skipped', []))}")


if __name__ == "__main__":
    main()
