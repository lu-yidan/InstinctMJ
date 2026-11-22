"""Actor-only LR override survives optimizer resume; other LRs are unchanged."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from run_stages import inherit_agent_config, override_actor_lr_scale


class ActorLrOverrideTests(unittest.TestCase):
    def test_override_follows_saved_config_and_only_changes_actor(self):
        agent = SimpleNamespace(lr=.0001, pi_lr_scale=1., planner_lr=.0001)
        inherit_agent_config(agent, dict(lr=.0002, pi_lr_scale=.5, planner_lr=.0001))
        override_actor_lr_scale(agent, .25)
        self.assertEqual(agent.pi_lr_scale, .25)
        self.assertEqual(agent.lr, .0002)
        self.assertEqual(agent.planner_lr, .0001)
        override_actor_lr_scale(agent, None)
        self.assertEqual(agent.pi_lr_scale, .25)

    def test_invalid_multiplier_is_rejected(self):
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                override_actor_lr_scale(SimpleNamespace(pi_lr_scale=1.), value)

    def test_resumed_muon_and_adam_keep_new_lr_and_old_moments(self):
        if not hasattr(torch.optim, 'Muon'):
            self.skipTest('torch.optim.Muon requires the training Torch runtime')
        path = Path(__file__).resolve().parents[2]/'src/instinct_mj/rl/acp/optim.py'
        spec = importlib.util.spec_from_file_location('acp_lr_optimizer', path)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        def network():
            return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ELU(),
                                       torch.nn.Linear(4, 4), torch.nn.ELU(), torch.nn.Linear(4, 2))
        actor = network()
        previous = api.MuonAdam([actor], lr=.0002)
        actor(torch.ones(2, 3)).sum().backward()
        previous.step()
        saved = previous.state_dict()
        resumed = api.MuonAdam([network()], lr=.00005)
        resumed.load_state_dict(saved)
        self.assertIsNotNone(resumed.muon)
        self.assertTrue(all(abs(float(g['lr'])-.00005) < 1e-10 for g in resumed.param_groups))
        state = resumed.state_dict()
        for optimizer in ('adam','muon'):
            self.assertEqual(saved[optimizer]['state'].keys(), state[optimizer]['state'].keys())
            for parameter, values in saved[optimizer]['state'].items():
                for key, value in values.items():
                    other = state[optimizer]['state'][parameter][key]
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, other), (optimizer,parameter,key))
                    else:
                        self.assertEqual(value, other)


if __name__ == '__main__':
    unittest.main()
