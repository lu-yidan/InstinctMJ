#!/usr/bin/env python3
"""One native Stage2 -> Stage3 flat run, capped at 2k; no automatic extension.

Hooks only audit the loaded state and publish receipts after a complete save.
The training algorithm, schedules, gradients and collection are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def check_transition(cfg, iteration, source_iteration):
    expected = dict(contact_source="planner", train_planner=True, restart_iteration=True,
        freeze_tracker=False, freeze_planner=False, handover_hold_iteration=None,
        wm_only_iters=0, vfs_warmup_iters=500, planner_seed_initial_support=False,
        max_iterations=2000, save_interval=1000)
    wrong = {key:{"expected":value,"actual":cfg.get(key)} for key,value in expected.items() if cfg.get(key) != value}
    if iteration != 0 or source_iteration != 9000 or wrong:
        raise ValueError("Not the authorized fresh Stage3 transition: " + repr((iteration, source_iteration, wrong)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from instinct_mj.rl.acp.stage3_runner import ACPStage3Runner
    import run_stages

    checkpoint = args.checkpoint.resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.name.replace("model_", "acp_state_"))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck["iter"] != 9000 or not ck.get("has_planner") or (ck.get("infos") or {}).get("stage") == 3:
        raise ValueError("This bounded experiment must start from the original native Stage2 9k")
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint, sidecar, checkpoint.parent/"params/agent.yaml")}
    expected_model = "9093e3e90e6cbc7f3368d5a5d39460680cb1dafa77736f0cbb26f40578af3b88"
    expected_sidecar = "0fae4bfd91da5d2a9976859a4b14bd263c701fc680bc4f69640e9d739102410d"
    if hashes[str(checkpoint)] != expected_model or hashes[str(sidecar)] != expected_sidecar:
        raise ValueError("Source hashes differ from the retained native 9k originals")
    original_setup, original_save = ACPStage3Runner.setup_training, ACPStage3Runner.save

    def setup(self):
        original_setup(self)
        check_transition(self.cfg, self.it, self.checkpoint_it)
        checked = 0
        for group,actual in (("model_state_dict",self.ac.state_dict()),
                             ("planner_state_dict",self.planner.state_dict()),
                             ("policy_normalizer_state_dict",self.normalizers["policy"].state_dict())):
            if actual.keys() != ck[group].keys():
                raise ValueError("Loaded model keys differ: " + group)
            for key,value in actual.items():
                if not torch.equal(value.cpu(),ck[group][key]):
                    raise ValueError("Loaded tensor differs: " + group + "." + key)
                checked += 1
        saved_sidecar = torch.load(sidecar, map_location="cpu", weights_only=False)
        for group,actual in (("wm",self.wm.state_dict()),("critic",self.critic.state_dict())):
            for key,value in actual.items():
                if not torch.equal(value.cpu(),saved_sidecar[group][key]):
                    raise ValueError("Loaded sidecar tensor differs: " + group + "." + key)
                checked += 1
        if "critic_goal" in saved_sidecar or self.ret_std != {"ref":None,"goal":None}:
            raise ValueError("Expected a fresh Stage3 goal critic/return scales")
        write_json(Path(self.log_dir)/"transition_verified.json", dict(source_hashes=hashes,
            stage3_start_iteration=self.it, source_iteration=self.checkpoint_it,
            equal_source_tensors=checked, effective_config=self.cfg,
            new_vfs=True, goal_actor_gradient_starts_at_iteration=500,
            note="Separate user-authorized downstream simulation trial. Formal Stage2 gate remains failed."))

    def save(self,path,infos=None):
        original_save(self,path,infos)
        model_path, sc_path = Path(path), Path(self._sidecar(path))
        saved = torch.load(model_path, map_location="cpu", weights_only=True)
        sc = torch.load(sc_path, map_location="cpu", weights_only=False)
        if saved["iter"] > 2000 or saved["infos"]["stage"] != 3 or sc["iteration"] != saved["iter"] or "critic_goal" not in sc:
            raise ValueError("Incomplete/invalid bounded Stage3 checkpoint")
        write_json(model_path.with_suffix(".ready.json"), dict(iteration=saved["iter"], stage=3,
            checkpoint=str(model_path), checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            sidecar_sha256=hashlib.sha256(sc_path.read_bytes()).hexdigest()))

    ACPStage3Runner.setup_training, ACPStage3Runner.save = setup, save
    sys.argv = [str(Path(run_stages.__file__)), "stage3", "--dataset", str(args.dataset.resolve()),
        "--run-dir", str(args.run_dir.resolve()), "--resume-checkpoint", str(checkpoint),
        "--iterations", "2000", "--gpu", str(args.gpu)]
    return run_stages.main()


if __name__ == "__main__":
    raise SystemExit(main())
