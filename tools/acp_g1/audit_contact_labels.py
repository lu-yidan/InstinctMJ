#!/usr/bin/env python3
"""Compare exact hindsight supervision with reference commands on saved rollouts.

No simulator, training, or checkpoint mutation. Reference commands are a semantic
comparison, not ground truth for the independently simulated robot's contacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def load_labeler():
    source = Path(__file__).resolve().parents[2] / "src/instinct_mj/rl/acp/hindsight.py"
    spec = importlib.util.spec_from_file_location("acp_audit_hindsight", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.label_episode, source


def segments(contact):
    """Half-open True segments, retaining boundary-censoring information."""
    contact = np.asarray(contact, dtype=bool)
    edges = np.diff(np.r_[False, contact, False].astype(np.int8))
    return [{"start": int(a), "end": int(b), "length": int(b-a),
             "left_censored": bool(a == 0), "right_censored": bool(b == len(contact))}
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def compare_targets(position, valid, reference, ref_valid, row_mask):
    """Report validity confusion and position errors only where both are valid."""
    a, b = np.asarray(valid)[row_mask], np.asarray(ref_valid)[row_mask]
    both = a & b
    delta = (np.asarray(position)[row_mask] - np.asarray(reference)[row_mask])[both]
    xy = np.linalg.norm(delta[:, :2], axis=-1)
    return {"body_steps": int(a.size), "both_valid": int(both.sum()),
            "hindsight_only_valid": int((a & ~b).sum()), "reference_only_valid": int((~a & b).sum()),
            "both_invalid": int((~a & ~b).sum()),
            "validity_agreement": float((a == b).mean()) if a.size else None,
            "xy_error_median_m": float(np.median(xy)) if len(xy) else None,
            "xy_error_p90_m": float(np.quantile(xy, .9)) if len(xy) else None,
            "xy_error_gt_25cm": int((xy > .25).sum()),
            "signed_xyz_mean_m": delta.mean(0).tolist() if len(delta) else None}


def match_onsets(reference, measured, max_ticks=10):
    """One-to-one nearest event matching; the tolerance is diagnostic only."""
    candidates = sorted((abs(a-b), a, b) for a in reference for b in measured if abs(a-b) <= max_ticks)
    ref_used, meas_used, pairs = set(), set(), []
    for _, a, b in candidates:
        if a not in ref_used and b not in meas_used:
            pairs.append((a, b))
            ref_used.add(a)
            meas_used.add(b)
    return sorted(pairs)


def geometry_pairs(rows, ids, body, reference_runs, force_runs, min_segment):
    """Decompose contact-point mismatch into link tracking and surface offsets."""
    if "training_contact_link_pos_w" not in rows:
        return []
    pairs = match_onsets([x["start"] for x in reference_runs if not x["left_censored"]],
                         [x["start"] for x in force_runs if not x["left_censored"] and x["length"] >= min_segment])
    result = []
    for r, a in pairs:
        prev, rr, ar = ids[r-1], ids[r], ids[a]
        if not rows["ref_valid"][prev, body]:
            continue
        # The preceding reference command points to this onset, in its own
        # estimated pelvis frame. Do not transform it using the true pelvis.
        yaw = rows["training_estimated_pelvis_yaw"][prev]
        c, s = np.cos(yaw), np.sin(yaw)
        rotate = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        ref_cp = rotate @ rows["ref_targets_local"][prev, body] + rows["training_estimated_pelvis_pos_w"][prev]
        actual_cp = rows["training_contact_pos_w"][ar, body]
        actual_link = rows["training_contact_link_pos_w"][ar, body]
        ref_link = rows["reference_contact_link_pos_w"][rr, body]
        actual_rot = rows["training_contact_link_rot_w"][ar, body]
        ref_rot = rows["reference_contact_link_rot_w"][rr, body]
        tracking = rotate.T @ (actual_link-ref_link)
        surface = rotate.T @ ((actual_cp-actual_link)-(ref_cp-ref_link))
        total = rotate.T @ (actual_cp-ref_cp)
        if not np.allclose(total, tracking+surface, atol=1e-6):
            raise ValueError("Contact error decomposition does not reconstruct total")
        result.append({"reference_tick": int(r), "force_tick": int(a), "lag_ticks": int(a-r),
                       "contact_delta_yaw_m": total.tolist(), "link_tracking_delta_yaw_m": tracking.tolist(),
                       "surface_offset_delta_yaw_m": surface.tolist(),
                       "actual_contact_offset_link_m": (actual_rot.T @ (actual_cp-actual_link)).tolist(),
                       "reference_contact_offset_link_m": (ref_rot.T @ (ref_cp-ref_link)).tolist()})
    return result


def analyze(rows, schema, variants=(1,), min_segment=10, horizon=75):
    label_episode, _ = load_labeler()
    output_rows, records, event_rows, geometry = [], {}, [], []
    for variant in variants:
        variant_mask = rows["variant"] == variant
        for clip in np.unique(rows["clip_index"][variant_mask]):
            clip_mask = variant_mask & (rows["clip_index"] == clip)
            for episode in np.unique(rows["episode"][clip_mask]):
                ids = np.flatnonzero(clip_mask & (rows["episode"] == episode))
                ids = ids[np.argsort(rows["tick"][ids])]
                ticks = rows["tick"][ids]
                if ticks[0] != 0 or not np.array_equal(ticks, np.arange(len(ids))):
                    raise ValueError("Hindsight requires a contiguous episode starting at reset")
                if not bool(rows["done_after_step"][ids[-1]]):
                    raise ValueError("Refusing incomplete episode: future contact labels would be censored")
                common = [torch.as_tensor(rows[key][ids]) for key in (
                    "training_contact_force_z", "training_contact_pos_w",
                    "training_estimated_pelvis_pos_w", "training_estimated_pelvis_yaw")]
                ref_flags = rows["reference_contact_flags"][ids].astype(bool)
                for mode in ("original", "seed_initial_support"):
                    pos, flag = label_episode(*common, min_segment=min_segment, horizon=horizon,
                        reference_contacts=torch.as_tensor(ref_flags) if mode == "seed_initial_support" else None)
                    values = {"variant": np.full(len(ids), variant, dtype=np.int8),
                              "clip_index": np.full(len(ids), clip), "episode": np.full(len(ids), episode),
                              "tick": ticks, "mode": np.full(len(ids), mode),
                              "paper_success": np.full(len(ids), bool(rows["success_after_step"][ids[-1]])),
                              "target_pos": pos.numpy(), "target_valid": flag.numpy(),
                              "reference_pos": rows["ref_targets_local"][ids], "reference_valid": rows["ref_valid"][ids],
                              "in_interior": (ticks >= 25) & (ticks + horizon < len(ids))}
                    for key, value in values.items():
                        records.setdefault(key, []).append(value)
                for body in (3, 4):
                    ref_runs = segments(ref_flags[:, body])
                    force_runs = segments(np.abs(rows["training_contact_force_z"][ids, body]) > 1.)
                    event_rows.append({"variant": int(variant), "clip": schema["clip_names"][int(clip)],
                                       "episode": int(episode), "body": schema["contact_body_names"][body],
                                       "reference_segments": ref_runs, "force_segments": force_runs})
                    geometry.extend({"variant": int(variant), "clip": schema["clip_names"][int(clip)],
                                     "episode": int(episode), "body": schema["contact_body_names"][body], **pair}
                                    for pair in geometry_pairs(rows, ids, body, ref_runs, force_runs, min_segment))
    records = {key: np.concatenate(value) for key, value in records.items()}
    for variant in variants:
        for mode in ("original", "seed_initial_support"):
            scope = (records["variant"] == variant) & (records["mode"] == mode)
            for part, mask in (("all", scope), ("interior_after_0.5s_and_full_future", scope & records["in_interior"]),
                               ("paper_successful_interior", scope & records["in_interior"] & records["paper_success"]),
                               ("first_0.5s", scope & (records["tick"] < 25))):
                for body_group, bodies in (("feet", [3, 4]), ("all_bodies", list(range(9)))):
                    metrics = compare_targets(records["target_pos"][:, bodies], records["target_valid"][:, bodies],
                                              records["reference_pos"][:, bodies], records["reference_valid"][:, bodies], mask)
                    output_rows.append({"variant": int(variant), "mode": mode, "period": part,
                                        "bodies": body_group, **metrics})
    complete_ref = [run for ep in event_rows for run in ep["reference_segments"]
                    if not run["left_censored"] and not run["right_censored"]]
    complete_force = [run for ep in event_rows for run in ep["force_segments"]
                      if not run["left_censored"] and not run["right_censored"]]
    summary = {"rows": output_rows, "episodes": len(event_rows)//2,
               "complete_foot_contact_segments": {
                   "reference": len(complete_ref), "reference_shorter_than_min_segment": sum(x["length"] < min_segment for x in complete_ref),
                   "force": len(complete_force), "force_shorter_than_min_segment": sum(x["length"] < min_segment for x in complete_force)},
               "events": event_rows, "matched_onset_geometry": geometry,
               "geometry_summary": {"pairs": len(geometry), "matching_tolerance_ticks": 10,
                   "mean_vectors_m": {key: np.mean([x[key] for x in geometry], axis=0).tolist()
                       for key in ("contact_delta_yaw_m", "link_tracking_delta_yaw_m", "surface_offset_delta_yaw_m",
                                   "actual_contact_offset_link_m", "reference_contact_offset_link_m")} if geometry else {}}}
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="B", help="A, B, C or a comma-separated subset; B is reference bypass")
    parser.add_argument("--min-segment", type=int, default=10)
    args = parser.parse_args()
    variants = tuple("ABC".index(x) for x in args.variants.split(","))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = args.run_dir/"trace_selected.npz"
    schema = json.loads((args.run_dir/"trace_selected.schema.json").read_text())
    with np.load(trace) as archive:
        needed = ("variant", "clip_index", "episode", "tick", "done_after_step", "success_after_step",
                  "training_contact_force_z", "training_contact_pos_w", "training_estimated_pelvis_pos_w",
                  "training_estimated_pelvis_yaw", "reference_contact_flags", "ref_targets_local", "ref_valid",
                  "training_contact_link_pos_w", "training_contact_link_rot_w",
                  "reference_contact_link_pos_w", "reference_contact_link_rot_w")
        # NPZ entries are lazy: materialize once, rather than decompressing a
        # whole dataset field separately for each episode. Raw 939-D inputs
        # are deliberately not loaded for this geometry-only audit.
        rows = {key: archive[key] for key in needed if key in archive}
    summary, arrays = analyze(rows, schema, variants, args.min_segment)
    _, label_source = load_labeler()
    summary.update(trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        labeler_sha256=hashlib.sha256(label_source.read_bytes()).hexdigest(),
        min_segment=args.min_segment, horizon=75, force_threshold_N=1.,
        caveats=["Paper-rollout supervision audit; does not measure the DR training replay distribution.",
                 "Reference motion contacts and simulated hindsight contacts need not coincide.",
                 "Foot contact segment counts exclude boundary-censored segments; label comparison includes them.",
                 "Interior excludes first 0.5 s and requires a complete recorded 1.5 s future.",
                 "Paper-successful interior is separated from all completed episodes; paper success is not the training replay's success filter.",
                 "Position errors are measured only when both targets are valid; invalid zero vectors are not positions."])
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(args.output_dir/"labels.npz", **arrays)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("events", "rows", "matched_onset_geometry")}, indent=2))
    for row in summary["rows"]:
        if row["bodies"] == "feet":
            print(json.dumps(row))


if __name__ == "__main__":
    main()
