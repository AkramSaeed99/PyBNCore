import random
import unittest
from pathlib import Path

import numpy as np
from pybncore import BatchExecutionEngine, Graph, JunctionTreeCompiler
from pybncore.wrapper import PyBNCoreWrapper


NET_050 = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "net_050n.xdsl"
)


class BatchUQRegressionTests(unittest.TestCase):
    def test_batch_query_parity_against_scalar_loop(self) -> None:
        w = PyBNCoreWrapper(str(NET_050))
        nodes = list(w._name_to_id.keys())
        q_nodes = nodes[:8]

        rng = random.Random(7)
        scenarios = []
        for _ in range(16):
            ev = {}
            for n in rng.sample(nodes, 4):
                outs = w.get_outcomes(n)
                ev[n] = outs[rng.randrange(len(outs))]
            scenarios.append(ev)

        scalar = []
        for ev in scenarios:
            w.clear_evidence()
            w.set_evidence(ev)
            scalar.append(w.batch_query_marginals(q_nodes))

        e_mat = np.full((len(scenarios), len(nodes)), -1, dtype=np.int32)
        for i, ev in enumerate(scenarios):
            for n, s in ev.items():
                e_mat[i, w._name_to_id[n]] = w._node_states[n].index(s)

        batched = w.batch_query_marginals(q_nodes, evidence_matrix=e_mat)

        max_diff = 0.0
        for i in range(len(scenarios)):
            for qn in q_nodes:
                arr = np.asarray(batched[qn], dtype=float)[i]
                ref = np.array(
                    [scalar[i][qn][st] for st in w.get_outcomes(qn)],
                    dtype=float,
                )
                max_diff = max(max_diff, float(np.max(np.abs(arr - ref))))

        self.assertLessEqual(max_diff, 1e-8, f"max_diff={max_diff}")

    def test_set_cpt_batched_chunked_matches_scalar(self) -> None:
        w = PyBNCoreWrapper()
        w._chunk_size = 2  # force multi-chunk execution for B=5
        w.load(str(NET_050))

        nodes = w.nodes()
        root = next(
            n for n in nodes if len(w.parents(n)) == 0 and len(w.get_outcomes(n)) >= 2
        )
        children = w.children(root)
        self.assertTrue(children, "Expected a child for selected root node.")
        q_node = children[0]

        base = w.get_cpt_shaped(root)
        rows, states = base.shape
        self.assertEqual(states, 2, "This regression test expects binary root states.")

        bsz = 5
        cpt_batch = np.zeros((rows, states, bsz), dtype=float)
        for b in range(bsz):
            p = 0.1 + 0.2 * b
            cpt_batch[:, 0, b] = p
            cpt_batch[:, 1, b] = 1.0 - p

        w.set_cpt_batched(root, cpt_batch, validate=True)
        e_mat = np.full((bsz, len(w._name_to_id)), -1, dtype=np.int32)
        out_batch = np.asarray(
            w.batch_query_marginals([q_node], evidence_matrix=e_mat)[q_node],
            dtype=float,
        )

        out_scalar = []
        for b in range(bsz):
            w.set_cpt(root, cpt_batch[:, :, b], validate=True)
            w.clear_evidence()
            pmf = w.batch_query_marginals([q_node])[q_node]
            out_scalar.append(np.array([pmf[s] for s in w.get_outcomes(q_node)]))
        out_scalar = np.vstack(out_scalar)

        max_diff = float(np.max(np.abs(out_batch - out_scalar)))
        self.assertLessEqual(max_diff, 1e-8, f"max_diff={max_diff}")

    def test_inconsistent_batch_row_returns_zero_row(self) -> None:
        g = Graph()
        a = g.add_variable("A", ["0", "1"])
        b = g.add_variable("B", ["0", "1"])
        c = g.add_variable("C", ["0", "1"])
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        g.set_cpt("A", np.array([1.0, 0.0], dtype=np.float64))
        g.set_cpt("B", np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64))
        g.set_cpt("C", np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64))

        jt = JunctionTreeCompiler.compile(g, "min_fill")
        eng = BatchExecutionEngine(jt, 1, 8)

        ev = np.full((2, 3), -1, dtype=np.int32)
        ev[0, a] = 0
        ev[0, c] = 0
        ev[1, a] = 0
        ev[1, c] = 1  # impossible with deterministic chain

        q = np.array([b], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        out = np.zeros((2, 2), dtype=np.float64)
        eng.evaluate_multi(ev, out, q, offsets)

        self.assertAlmostEqual(float(np.sum(out[0])), 1.0, places=12)
        self.assertTrue(np.allclose(out[1], 0.0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
