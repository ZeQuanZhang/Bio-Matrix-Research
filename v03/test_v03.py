"""Small deterministic implementation tests; CPU here is intentional."""
import math
import unittest
import numpy as np
import torch
from .data import example, solve, collate
from .graph import graph, statistics
from .model import Config, Network, rotate


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.batch = collate([example(np.random.default_rng(i), 8, 2) for i in range(2)])

    def model(self, kind='C3'):
        torch.manual_seed(21)
        return Network(Config(model=kind, nodes=32, width=32, bottleneck=8, iterations=3))

    def call(self, model, **kw):
        return model(**{k: v for k, v in self.batch.items() if k != 'target'}, **kw)

    def test_sparse_graphs(self):
        for n in (32, 128, 512):
            for topology in ('connected', 'cut'):
                g = graph(n, topology=topology)
                self.assertEqual(g.shape, (n, 8))
                self.assertEqual(statistics(g)['components'], 2 if topology == 'cut' else 1)
                for i, row in enumerate(g.tolist()):
                    self.assertEqual(len(set(row)), 8); self.assertNotIn(i, row)
                    for j in row:
                        self.assertIn(i, g[j].tolist())

    def test_data_solver_and_permutations(self):
        rng = np.random.default_rng(81)
        for family in ('retrieve', 'trace', 'compare', 'count', 'rule'):
            for _ in range(20):
                e = example(rng, 32, 3, .8, family)
                self.assertEqual(e['target'], solve(e['world'], e['spec']))
                self.assertNotIn('target', {k for k in collate([e]) if k != 'target'})
        m = self.model(); original = self.call(m)[0]
        shuffled = {k: v.clone() for k, v in self.batch.items()}
        shuffled['records'] = shuffled['records'].flip(1); shuffled['mask'] = shuffled['mask'].flip(1)
        changed = m(**{k: v for k, v in shuffled.items() if k != 'target'})[0]
        torch.testing.assert_close(original, changed, atol=1e-6, rtol=1e-5)

    def test_angle_rotation_equivalence(self):
        theta = torch.linspace(-math.pi, math.pi, 32, dtype=torch.float64)
        z = torch.stack((theta.cos(), theta.sin()), -1)
        delta = torch.linspace(-.2, .2, 32, dtype=torch.float64)
        expected = torch.stack(((theta+delta).cos(), (theta+delta).sin()), -1)
        torch.testing.assert_close(rotate(z, delta), expected, atol=1e-12, rtol=1e-12)

    def test_unit_circle_and_active_gradients(self):
        for kind in ('C1', 'C2', 'C3'):
            m = self.model(kind); logits, (s, z, _) = self.call(m)
            torch.nn.functional.cross_entropy(logits, self.batch['target']).backward()
            for name, p in m.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(bool(p.grad.isfinite().all()), name)
            self.assertGreater(float(s.detach().std(1).mean()), 1e-4)
            self.assertGreater(float((s.detach()[:, 0]-s.detach()[:, 1]).abs().mean()), 1e-4)
            if kind != 'C1':
                torch.testing.assert_close(z.square().sum(-1), torch.ones_like(z[..., 0]), atol=2e-6, rtol=0)

    def test_common_rotation_and_node_renaming(self):
        m = self.model(); original = self.call(m)[0]
        with torch.no_grad():
            m.initial_z.copy_(rotate(m.initial_z, m.initial_z.new_full((32,), .73)))
        torch.testing.assert_close(original, self.call(m)[0], atol=1e-6, rtol=1e-5)
        permutation = torch.randperm(32); inverse = permutation.argsort()
        with torch.no_grad():
            m.roles = m.roles[permutation]; m.initial_z = m.initial_z[permutation]
            m.neighbors = inverse[m.neighbors[permutation]]
        torch.testing.assert_close(original, self.call(m)[0], atol=1e-6, rtol=1e-5)

    def test_factorized_readout(self):
        m = self.model(); h = torch.randn(2, m.cfg.width); g = torch.rand(2, m.cfg.rank)
        factored = m.u(g*m.v(h))
        dense = (m.u.weight[None] @ torch.diag_embed(g) @ m.v.weight[None] @ h[..., None]).squeeze(-1)
        torch.testing.assert_close(factored, dense)

    def test_two_cluster_positive_control(self):
        theta = torch.cat((torch.linspace(-.5, .5, 8), torch.linspace(2.5, 3.5, 8)))
        z = torch.stack((theta.cos(), theta.sin()), -1)
        before = torch.stack((z[:8].mean(0).norm(), z[8:].mean(0).norm())).mean()
        for _ in range(60):
            cross = z[:, None, 0]*z[None, :, 1]-z[:, None, 1]*z[None, :, 0]
            same = torch.arange(16)[:, None]//8 == torch.arange(16)[None, :]//8
            z = rotate(z, .1*(cross*same).sum(-1)/8)
        after = torch.stack((z[:8].mean(0).norm(), z[8:].mean(0).norm())).mean()
        self.assertGreater(float(after), .999); self.assertGreater(float(after), float(before))


if __name__ == '__main__':
    unittest.main()
