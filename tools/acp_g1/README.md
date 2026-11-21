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
