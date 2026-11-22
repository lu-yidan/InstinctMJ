"""Render frozen-policy diagnostic traces; no policy or simulator is loaded."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    schema = json.loads((root/'trace_selected.schema.json').read_text())
    events = json.loads((root/'first_divergence.json').read_text())
    trace_path = root/'trace_selected.npz'
    with np.load(trace_path, allow_pickle=False) as archive:
        rows = {key: archive[key] for key in archive.files}
    out = root/'figures'
    out.mkdir(exist_ok=True)
    labels = ['A: Stage 1 / reference', 'B: Stage 2 / reference', 'C: Stage 2 / planner']
    colors = ['#2878b5', '#2a9d64', '#d84a42']
    artifacts = []
    for clip in np.unique(rows['clip_index']):
        name = schema['clip_names'][int(clip)]
        base = (rows['clip_index'] == clip) & (rows['episode'] == args.episode)
        by_variant = []
        for variant in range(3):
            ids = np.flatnonzero(base & (rows['variant'] == variant))
            ids = ids[np.argsort(rows['tick'][ids])]
            if not len(ids):
                raise ValueError(f'Missing variant {variant}: {name} episode {args.episode}')
            by_variant.append(ids)
        event = next(row for row in events if row['clip_index'] == int(clip) and row['episode'] == args.episode)
        tick = event['first_clear_pelvis_xy_difference_gt_5cm_sustained_5ticks']
        dt = .02
        for ids in by_variant:
            if len(ids) > 1:
                dt = float(np.median(np.diff(rows['time_s'][ids])))
                break
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        ref_ids = by_variant[0]
        ref = rows['ref_pelvis_pos_w'][ref_ids]
        axes[0].plot(ref[:, 0], ref[:, 1], color='#666666', linestyle='--', linewidth=1.5, label='Reference pelvis')
        for variant, ids in enumerate(by_variant):
            pos = rows['pelvis_pos_w'][ids]
            axes[0].plot(pos[:, 0], pos[:, 1], color=colors[variant], label=labels[variant])
            axes[0].scatter(pos[-1, 0], pos[-1, 1], color=colors[variant], s=24)
            error = np.linalg.norm(pos[:, :2]-rows['ref_pelvis_pos_w'][ids, :2], axis=1)
            axes[1].plot(rows['time_s'][ids], error, color=colors[variant], label=labels[variant])
        axes[0].set(xlabel='World x (m)', ylabel='World y (m)', title='Closed-loop pelvis trajectory')
        axes[0].set_aspect('equal', adjustable='datalim')
        axes[0].legend(fontsize=8)
        axes[1].axhline(.5, color='#888888', linestyle=':', linewidth=1, label='Endpoint XY threshold: 0.5 m')
        axes[1].set(xlabel='Time (s)', ylabel='Horizontal error (m)', title='Error to reference pelvis')
        axes[1].legend(fontsize=8)
        ids = by_variant[2]
        foot_error = rows['foot_target_error_xyz'][ids]
        for foot, color, side in [(0, '#9467bd', 'Left'), (1, '#d48b14', 'Right')]:
            axes[2].plot(rows['time_s'][ids], foot_error[:, foot, 0], color=color, label=side+' forward')
            axes[2].plot(rows['time_s'][ids], foot_error[:, foot, 1], color=color, linestyle='--', label=side+' lateral')
        axes[2].axhline(0, color='#888888', linewidth=.7)
        axes[2].set(xlabel='Time (s)', ylabel='Planner minus reference (m)', title='Same-state foot target error in C')
        axes[2].set_xlim(axes[1].get_xlim())
        axes[2].legend(fontsize=8)
        if tick is not None:
            for ax in axes[1:]:
                ax.axvline(tick*dt, color='#222222', linestyle=':', linewidth=1.3)
            cycle = event['measured_contact_cycle_at_clear_divergence']['C_left']
            if cycle['complete_onset_to_onset']:
                for ax in axes[1:]:
                    ax.axvspan(cycle['cycle_start_tick']*dt, cycle['cycle_end_tick']*dt,
                               color='#bbbbbb', alpha=.15)
        goal_tick = event['first_world_goal_difference_tick']
        if goal_tick is not None:
            for ax in axes[1:]:
                ax.axvline(goal_tick*dt, color='#b05c96', linestyle='--', linewidth=1,
                           label='B/C world goals first differ')
        axes[1].legend(fontsize=8)
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        title = f'{name} | episode {args.episode} | frozen weights, paper settings'
        if tick is not None:
            title += f' | B/C separation >5 cm for 5 frames begins at {tick*dt:.2f} s'
        fig.suptitle(title, fontsize=12)
        path = out/f'{name}_episode{args.episode}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        artifacts.append({'clip':name, 'episode':args.episode, 'figure':str(path),
                          'clear_divergence_tick':tick, 'control_dt':dt})
    manifest = {'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                'episode_selection':'fixed CLI episode, not selected by outcome',
                'missing_target_error':'NaN gaps mean either reference or planner target is invalid',
                'gray_interval':'C left-foot complete measured onset-to-onset cycle containing clear divergence',
                'artifacts':artifacts}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
