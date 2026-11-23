"""Resume warmup gates actual policy updates without resetting ACP clocks."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from run_stages import override_resume_wm_only

source=Path(__file__).resolve().parents[2]/'src/instinct_mj/rl/acp/runner.py'
tree=ast.parse(source.read_text())
runner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ACPRunner')
learn=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='learn')
gate=next(n.test for n in ast.walk(learn) if isinstance(n,ast.If)
          and 'wm_only_iters' in ast.unparse(n.test))
policy_gate=compile(ast.Expression(body=gate),str(source),'eval')

class ResumeWmOnlyTests(unittest.TestCase):
    def test_existing_runner_gate_holds_200_then_updates_500_without_changing_eta(self):
        agent=SimpleNamespace(wm_only_iters=500,eta_obs_start_iter=1000,eta_rew_start_iter=3000)
        override_resume_wm_only(agent,9000,200)
        self.assertEqual(agent.wm_only_iters,9200)
        cfg=dict(vars(agent),freeze_tracker=False)
        decisions=[eval(policy_gate,{'it':it,'c':cfg}) for it in range(9000,9700)]
        self.assertEqual(decisions[:200],[False]*200)
        self.assertEqual(decisions[200:],[True]*500)
        self.assertEqual((agent.eta_obs_start_iter,agent.eta_rew_start_iter),(1000,3000))
        cfg['freeze_tracker']=True
        self.assertFalse(eval(policy_gate,{'it':9500,'c':cfg}))

    def test_default_and_existing_longer_warmup_are_preserved(self):
        agent=SimpleNamespace(wm_only_iters=500)
        override_resume_wm_only(agent,None,None)
        self.assertEqual(agent.wm_only_iters,500)
        override_resume_wm_only(agent,9000,0)
        self.assertEqual(agent.wm_only_iters,9000)
        agent.wm_only_iters=9500
        override_resume_wm_only(agent,9000,200)
        self.assertEqual(agent.wm_only_iters,9500)

    def test_invalid_values_do_not_mutate_configuration(self):
        agent=SimpleNamespace(wm_only_iters=500)
        for iteration,value in ((None,200),(-1,200),(9000,-1),(9000,True),(9000,1.5)):
            with self.assertRaises(ValueError):override_resume_wm_only(agent,iteration,value)
            self.assertEqual(agent.wm_only_iters,500)

if __name__=='__main__':unittest.main()
