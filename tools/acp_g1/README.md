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

To continue Stage 1 after a failed evaluation, keep the old state and run
directories intact. Copy the configuration, choose a new `state_dir` and a new
empty stage `1` `run_dir`, and set stage `1` `resume_checkpoint` to the previous
`model_N.pt`. Increase its `iterations` to the total target, for example 45,000
when resuming iteration 30,000. The launcher restores the matching ACP training
sidecar and saved agent configuration; the new controller reruns all four gates
with the same thresholds. Use new post-export report paths as well. Changing
the total target does not make a failed old evaluation pass.

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

When moving runs to another host, copy the checkpoint, matching
`acp_state_N.pt`, and `params/agent.yaml` together, preserving their hashes.
`run_stages.py` and `diagnose_abc.py` accept `--heightmap-cache-dir` or
`ACP_HEIGHTMAP_CACHE_DIR` to replace the old host's cache path. The default is
`~/.cache/instinct_mj/supermimic_heightmaps`; this path is infrastructure, not a
learner setting. Keep hardware-specific repeats separate, and train both arms
of a controlled comparison on the same host.

If the policy update exceeds GPU memory, `run_stages.py --pi-batch 1024`
overrides only the imagined-start batch after restoring checkpoint settings.
Environment count, world-model/planner batches, replay and handover schedules
retain their existing values. This changes policy gradient sampling; apply the
same batch to both experimental arms and evaluate the resulting models anew.
Omitting the flag preserves the checkpoint's batch. The launch manifest records
the policy, world-model and planner batches explicitly.

When Torch's unused reserved blocks leave insufficient device memory for external
MuJoCo/Warp graph launches, `--release-cuda-cache` returns unused allocator blocks
between iterations, outside all captured updates. It retains live tensors and
graph pools and adds allocator overhead. Use the same setting in both arms and
stress-test past replay filling and actual planner updates; a ten-update policy
preflight alone does not cover that peak.

## Frozen Stage 1 / Stage 2 diagnosis

`diagnose_abc.py` compares three frozen policies, each with its checkpoint's own
saved normalizer: A is Stage 1 with reference contacts; B is Stage 2 with an
explicit `actor(normalizer(reference_observation))` call that bypasses planner
replacement; C is Stage 2's native inference policy with full planner takeover.
There is no learning, optimizer resume, normalizer update, or hardware control.

```bash
python tools/acp_g1/diagnose_abc.py \
  --dataset /absolute/path/php-flat \
  --stage1-checkpoint /absolute/path/stage1/model_45000.pt \
  --stage2-checkpoint /absolute/path/stage2/model_10000.pt \
  --output-dir /absolute/path/new-empty-abc \
  --num-envs 128 --episodes 8 --seed 42 --gpu 1
python -m unittest discover -s tools/acp_g1 -p test_diagnose_abc.py -v
python tools/acp_g1/plot_diagnosis.py --run-dir /absolute/path/new-empty-abc --episode 0
```

The default scope is every window in the training metadata. For the small
three-window diagnostic, add
`--clip-regex '^(php_cfg0080_01|php_cfg0098_00|php_cfg0002_00)$' --num-envs 24`.
That subset is not a full-dataset or held-out performance measurement. The
three named windows are traced by default; `--trace-clips` changes that list.
Output directories must be new or empty.

The plotting command uses Matplotlib and the saved trace only. It plots a fixed
episode's trajectories, contact target errors, measured contact cycle and first
world-goal divergence; `figures/manifest.json` records the input trace hash.

Protocol 2 preserves paper evaluation settings and native goal updates. Each
batch uses the same reset RNG and verifies identical actual initial physics,
motion frame/origin, world goal, raw observations, control history and solver
warmstart. Small independent-loader velocity rounding differences are recorded;
B/C receive A's actual initial state and rebuild their histories without
advancing reference time or consuming environment RNG. A itself is unchanged.
CPU/CUDA RNG states must be bitwise equal after reset. Policy-only draws,
including C's unused eta=1 Bernoulli draws, are isolated from environment RNG.
Afterward the closed loops run independently: native failures/resets can change
later world-goal sampling, which is recorded, and local goals always use each
robot's own pose. No common local-goal vector or external goal schedule is forced.

`--trace-training-contacts` additionally records the exact force-head hindsight
inputs: solver force history, contact points with the training fallback and the
estimated pelvis pose. This opt-in trace currently requires `force_head` mode.
It also saves simulated/reference contact-link poses, allowing solver contact
point offsets to be distinguished from link tracking error. `--trace-all-clips
--trace-variants B` saves every reference-bypass cohort without saving A/C traces;
all three policies are still evaluated. `--trace-planner-inputs` additionally
saves raw proprioception and points for offline, same-state planner comparisons.

`audit_contact_labels.py --run-dir /absolute/path/traced-abc --output-dir
/absolute/path/new-empty-label-audit` applies the actual hindsight labeler to
the saved episodes. It compares original and initial-support-corrected labels
with reference commands, preserving validity masks, boundary-censored contact
segments and a separate interior with a full future horizon. It refuses partial
episodes. The default is B only. This is an audit of recorded paper rollouts,
not a measurement of the DR training replay distribution; reference contacts
need not equal contacts of the independently simulated robot.

`A.paper.json`, `B.paper.json`, `C.paper.json` and `source_macros.json` separate
source-macro success from the mean over windows. `pairing.json` records strict
initial-state/RNG checks and the first later world-goal divergences.
`initial_states.npz` stores the actual initial tensors. `trace_selected.npz` and
its schema contain world pelvis/reference trajectories, same-state reference
and planner contacts/validity, signed foot target errors, own/local and shared
world goal information, actions, and same-state counterfactual actions.
`first_divergence.json` distinguishes numerical action differences from a pelvis
XY difference above 5 cm for five consecutive ticks and brackets the latter by
measured left/right foot contact rising edges. These are diagnostic thresholds,
not new policy gates. Invalid contact-position differences remain NaN with
explicit validity masks.

These reports explain failures; they do not replace a formal pipeline gate or
turn an old failed evaluation into a pass. First reproduce A using the native
`eval_tracker` CLI when comparing against a passed baseline. When evaluating
intermediate Stage 2 weights, C always forces full planner takeover even before
the training eta schedule reaches one; record training eta separately. A/B
differences concern the actor and its own normalizer together, not the actor
alone. Interpret closed-loop attribution alongside recorded goal divergence and
same-state counterfactual traces.

## Controlled planner refit

For a Stage 2 checkpoint whose reference bypass still works, `run_stages.py
stage2 --planner-refit` freezes the actor and its policy normalizer, collects with
reference contacts, and continues planner/world-model learning from the matching
checkpoint and ACP sidecar. Policy/critic updates are skipped. The final actor
and normalizer tensors (including counts) must be bitwise unchanged, or the run
fails; the receipt is `tracker_freeze_check.json`. This opt-in mode is not
inherited silently by subsequent ordinary runs. `--iterations` remains the total
iteration target, for example 4000 when resuming at 3000 for 1000 more updates.
Always evaluate C with full planner takeover afterward. No formal gate or stage
advance follows from this experimental mode by itself.

`--seed-initial-support` enables an experimental reset-label correction for
reference episodes. Their first sustained force contact, when it occurs during
the reference's initial support, joins the initial contact segment. The next-new
contact label then selects the following landing. A foot initially swinging or
landing after the initial reference liftoff keeps its first genuine contact.
Goal-episode labels retain their original behavior. This option is explicit for
each run and is recorded in the launch manifest.

`--actor-lr-scale 0.25` overrides only the actor's `pi_lr_scale`, after the saved
configuration is restored. With `lr=0.0002`, this gives actor LR `0.00005` while
the world model, critic, planner, handover schedules and labeling retain their
existing settings. The Muon/Adam optimizer restores its moments but keeps the
constructor's LR, including its device tensor for captured updates. Both the
multiplier and effective actor LR are recorded in the launch manifest. Use an
independent resumed control run at scale 1 when testing this change, because
unsaved replay buffers refill on resume.
