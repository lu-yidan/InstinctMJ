#!/usr/bin/env python3
"""Check known physical velocities at motion boundaries and interior (CPU)."""
from pathlib import Path

import torch
from instinct_mj.motion_reference import utils


def main():
    expected = Path(__file__).resolve().parents[2] / "src/instinct_mj/motion_reference/utils.py"
    assert Path(utils.__file__).resolve() == expected, utils.__file__
    dt = .02
    cases = 0
    for frames in (1, 2, 5):
        time = torch.arange(frames, dtype=torch.float64)[None, :, None] * dt
        for velocity in (torch.tensor([2.4, -.7, .15], dtype=torch.float64),
                         torch.linspace(-1.2, 1.2, 23, dtype=torch.float64)):
            positions = time * velocity
            result = utils.estimate_velocity(positions, dt)
            target = torch.zeros_like(result) if frames == 1 else velocity.expand_as(result)
            torch.testing.assert_close(result, target, atol=1e-9, rtol=0)
            cases += 1
        axis = torch.tensor([.2, -.4, .7], dtype=torch.float64)
        axis /= axis.norm()
        angle = time * .9
        quaternion = torch.cat((torch.cos(angle/2), torch.sin(angle/2)*axis), dim=-1)
        result = utils.estimate_angular_velocity(quaternion, dt)
        target = torch.zeros_like(result) if frames == 1 else (axis*.9).expand_as(result)
        torch.testing.assert_close(result, target, atol=1e-9, rtol=0)
        cases += 1
    # Nonconstant motion: centered interior, one-sided endpoints.
    time = torch.arange(5, dtype=torch.float64)[None, :, None] * dt
    expected = 2*time
    expected[:, 0] += dt
    expected[:, -1] -= dt
    torch.testing.assert_close(utils.estimate_velocity(time.square(), dt), expected, atol=1e-9, rtol=0)
    print(f"MOTION_VELOCITY_REGRESSION_PASS ({cases + 1} cases)")


if __name__ == "__main__":
    main()
