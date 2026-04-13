"""Loopy Belief Propagation for approximate inference.

This module provides an alternative inference engine for networks where exact
junction tree inference is intractable (treewidth > ~25). SMILE does not offer
approximate inference, making this a unique PyBNCore differentiator.

The implementation operates directly on the factor graph — no junction tree
compilation is needed. Messages are passed between variable and factor nodes
iteratively until convergence.
"""
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

from ._core import Graph


class LoopyBPEngine:
    """Loopy Belief Propagation engine for approximate marginal inference.

    Args:
        graph: The pybncore Graph object with CPTs assigned.
        damping: Message damping factor in [0, 1). 0 = no damping.
            Higher values slow convergence but improve stability.
        max_iterations: Maximum number of message passing iterations.
        tolerance: Convergence threshold on max message delta.
    """

    def __init__(
        self,
        graph: Graph,
        cpts: Dict[str, np.ndarray],
        damping: float = 0.5,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        self.graph = graph
        self.cpts = cpts
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self._num_vars = graph.num_variables()
        self._var_names: List[str] = []
        self._var_cards: List[int] = []
        self._name_to_id: Dict[str, int] = {}

        for i in range(self._num_vars):
            meta = graph.get_variable(i)
            self._var_names.append(meta.name)
            self._var_cards.append(len(meta.states))
            self._name_to_id[meta.name] = i

        # Build factor graph structure
        # Each variable has a factor (its CPT conditioned on parents)
        self._factors: List[dict] = []
        self._var_to_factors: List[List[int]] = [[] for _ in range(self._num_vars)]

        for i in range(self._num_vars):
            name = self._var_names[i]
            parents = list(graph.get_parents(i))
            family = parents + [i]  # parent ids + self

            # Build factor table from CPT
            cpt = cpts[name].copy()
            family_cards = [self._var_cards[v] for v in family]
            cpt = cpt.reshape(family_cards)

            factor_idx = len(self._factors)
            self._factors.append({
                'vars': family,
                'cards': family_cards,
                'table': cpt,
            })

            for v in family:
                self._var_to_factors[v].append(factor_idx)

    def infer(
        self,
        evidence: Optional[Dict[str, str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run loopy BP and return approximate marginals for all variables.

        Args:
            evidence: Dict mapping variable name to observed state name.

        Returns:
            Dict mapping variable name to marginal probability array.
        """
        n_factors = len(self._factors)

        # Initialize messages: var -> factor and factor -> var
        # msg_v2f[f][local_idx] = message from variable vars[local_idx] to factor f
        # msg_f2v[f][local_idx] = message from factor f to variable vars[local_idx]
        msg_v2f = []
        msg_f2v = []
        for f_idx, factor in enumerate(self._factors):
            n_vars_in_f = len(factor['vars'])
            msg_v2f.append([np.ones(self._var_cards[v]) / self._var_cards[v]
                           for v in factor['vars']])
            msg_f2v.append([np.ones(self._var_cards[v]) / self._var_cards[v]
                           for v in factor['vars']])

        # Process evidence: clamp variable beliefs
        evidence_mask = {}
        if evidence:
            for name, state_name in evidence.items():
                var_id = self._name_to_id[name]
                meta = self.graph.get_variable(var_id)
                states = list(meta.states)
                state_idx = states.index(state_name)
                ev = np.zeros(self._var_cards[var_id])
                ev[state_idx] = 1.0
                evidence_mask[var_id] = ev

        converged = False
        for iteration in range(self.max_iterations):
            max_delta = 0.0

            # --- Variable to Factor messages ---
            for f_idx, factor in enumerate(self._factors):
                for local_idx, var_id in enumerate(factor['vars']):
                    if var_id in evidence_mask:
                        new_msg = evidence_mask[var_id].copy()
                    else:
                        # Product of all incoming factor-to-var messages
                        # EXCEPT from factor f_idx
                        new_msg = np.ones(self._var_cards[var_id])
                        for other_f in self._var_to_factors[var_id]:
                            if other_f == f_idx:
                                continue
                            other_factor = self._factors[other_f]
                            other_local = other_factor['vars'].index(var_id)
                            new_msg *= msg_f2v[other_f][other_local]

                    # Normalize
                    s = new_msg.sum()
                    if s > 0:
                        new_msg /= s

                    # Damping
                    if self.damping > 0:
                        new_msg = (1 - self.damping) * new_msg + self.damping * msg_v2f[f_idx][local_idx]

                    delta = np.max(np.abs(new_msg - msg_v2f[f_idx][local_idx]))
                    max_delta = max(max_delta, delta)
                    msg_v2f[f_idx][local_idx] = new_msg

            # --- Factor to Variable messages ---
            for f_idx, factor in enumerate(self._factors):
                table = factor['table']
                n_vars_in_f = len(factor['vars'])

                for target_local, target_var in enumerate(factor['vars']):
                    # Multiply factor table by all incoming var-to-factor messages
                    # except from target_var, then marginalize over all but target_var
                    product = table.copy()
                    for local_idx in range(n_vars_in_f):
                        if local_idx == target_local:
                            continue
                        # Broadcast message along the appropriate axis
                        shape = [1] * n_vars_in_f
                        shape[local_idx] = self._var_cards[factor['vars'][local_idx]]
                        product *= msg_v2f[f_idx][local_idx].reshape(shape)

                    # Sum over all axes except target_local
                    axes = tuple(j for j in range(n_vars_in_f) if j != target_local)
                    new_msg = product.sum(axis=axes) if axes else product.flatten()

                    # Normalize
                    s = new_msg.sum()
                    if s > 0:
                        new_msg /= s

                    # Damping
                    if self.damping > 0:
                        new_msg = (1 - self.damping) * new_msg + self.damping * msg_f2v[f_idx][target_local]

                    delta = np.max(np.abs(new_msg - msg_f2v[f_idx][target_local]))
                    max_delta = max(max_delta, delta)
                    msg_f2v[f_idx][target_local] = new_msg

            if max_delta < self.tolerance:
                converged = True
                break

        # Compute beliefs (marginals) for each variable
        beliefs = {}
        for var_id in range(self._num_vars):
            name = self._var_names[var_id]

            if var_id in evidence_mask:
                beliefs[name] = evidence_mask[var_id].copy()
                continue

            belief = np.ones(self._var_cards[var_id])
            for f_idx in self._var_to_factors[var_id]:
                factor = self._factors[f_idx]
                local_idx = factor['vars'].index(var_id)
                belief *= msg_f2v[f_idx][local_idx]

            s = belief.sum()
            if s > 0:
                belief /= s
            beliefs[name] = belief

        return beliefs

    @property
    def converged(self) -> bool:
        """Whether the last inference call converged."""
        return getattr(self, '_converged', False)
