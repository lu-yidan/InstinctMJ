# ACP G1 data preparation

The first training subset comes from the user-selected [PHP training examples
release](https://github.com/amazon-far/php_parkour/releases/tag/training-motion-examples-v1).
It is a subset of example inputs, not the complete PHP parkour dataset.

```bash
python tools/acp_g1/build_php_dataset.py \
  --archive /absolute/path/php-motions-locomotion-v1.tar.gz \
  --output /absolute/path/new-empty-php-flat
python -m instinct_mj.scripts.acp.label_contacts \
  --dataset /absolute/path/new-empty-php-flat/all \
  --simplified-collision --workers 4
```

The archive SHA-256 is pinned in the builder. Without `--archive`, it downloads
that exact release asset into an adjacent `raw/` directory. `--license-file`
supplies a saved source license for offline conversion. Existing datasets are
never overwritten.

The released files are Holosoma motion NPZs at 50 Hz, with pelvis root
XYZ/WXYZ and 29 named joints. Only the 50 source pairs whose terrain NPY has
zero boxes are selected. The other 50 retain their original obstacle variants
in the archive and are excluded from this flat subset.

The builder selects the ACP asset's 23 joints by name and preserves continuous
4–6 second windows, including real standing, turning and stopping frames.
It recenters XY and applies one bounded constant Z translation per window using
the actual ACP foot capsules. It does not clip joints or synthesize trajectories
or labels. Windows with excessive remaining foot penetration are excluded.
All clips share one exact flat primitive terrain.

Original source-file hashes define the train/holdout split: 40 distinct sources
for training and 10 for holdout. Overlapping windows from a source always stay
in one split. `metadata.yaml` lists only training windows;
`holdout/metadata.yaml` points to the held-out windows. `metadata.all.yaml` and
`all/metadata.yaml` list the entire subset so official contact labels can be
generated once for every window without copying motions.

The prepared dataset has 110 training windows (32,609 frames) and 26 holdout
windows (7,717 frames). `provenance.json` records the archive, source-file
hashes, source frames, split, translation, rejected windows and FK statistics.
`contacts_validation.json` and `CONTACTS.sha256` record the completed official
label checks. Use the held-out metadata for evaluation and report it separately
from training clips.

## Simulation pipeline

`pipeline.py` defaults to printing a plan. `pipeline.example.json` describes
Stage 1 (30,000 iterations), Stage 2 (10,000), Stage 3 Flat (10,000), native
goal evaluation, bundle parity, replay, and deployment MuJoCo checks. It never
starts hardware control. Copy the example to the run area and check its absolute
paths before execution:

```bash
python tools/acp_g1/pipeline.py --config /absolute/path/pipeline.json
python tools/acp_g1/pipeline.py --config /absolute/path/pipeline.json --execute
```

`--execute --once` makes one scheduling pass, suitable for a monitor. Restarting
with the same config/state directory observes the child PID and exit receipt;
it does not duplicate a live or completed child. `status.json`, per-step logs,
process receipts, checkpoint hashes and evaluation reports remain in
`state_dir`. Configuration changes require a new state directory to prevent old
receipts from being applied silently to different work.

For an already-running Stage 1 launched outside the controller, put an `attach`
object in stage `1`, with its actual `pid`, `log`, and the absolute
`<run_dir>/exit_status.json` path as `exit_file`. A completed external run needs
the successful exit receipt. A checkpoint alone is insufficient evidence.

Each tracker stage is evaluated on GPU 1 in both paper and robust modes, on all
training and held-out windows separately. Gates use each original source's mean
window success, then average equally across original sources. Defaults require
train paper/robust success of 0.9/0.8 and holdout paper/robust success of 0.8/0.7.
Missing cohorts, overlapping source hashes, missing sidecars, nonzero exits or
failed thresholds stop the controller with `needs_attention`.

The evaluation script loads actor/planner/normalizer and builds only the live
state reader and analytic metrics. It does not resume the training world model
or optimizer; paper evaluation deliberately disables DR, whose context encoding
differs from training. Saved actor configurations, explicit seeds and metadata
hashes are retained in the evaluation path.
