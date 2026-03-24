import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ._core import Graph, JunctionTree, JunctionTreeCompiler, BatchExecutionEngine, VariableMetadata
from .io import read_xdsl

class PyBNCoreWrapper:
    """
    HCL-compatible wrapper integrating PyBNCore natively into legacy BDD_Engine ecosystems.
    Matches exact API specifications of SMILE backends (bn_smile.py / bn_pgmpy.py).
    """
    def __init__(self, model_path: Optional[str] = None):
        self._graph: Optional[Graph] = None
        self._jt: Optional[JunctionTree] = None
        self._engine: Optional[BatchExecutionEngine] = None
        
        self._node_names: List[str] = []
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        self._node_states: Dict[str, List[str]] = {}
        self._cpts: Dict[str, np.ndarray] = {}
        self._evidence: Dict[str, str] = {}
        
        self._is_compiled = False
        self._num_threads = 0
        self._chunk_size = 1024
        
        if model_path is not None:
            self.load(model_path)

    @classmethod
    def from_xdsl(cls, path: str) -> "PyBNCoreWrapper":
        return cls(path)

    def load(self, model_path: str) -> None:
        self._graph, self._cpts = read_xdsl(model_path)
        self._cache_metadata()
        self._compile()

    def _cache_metadata(self) -> None:
        self._node_names.clear()
        self._name_to_id.clear()
        self._id_to_name.clear()
        self._node_states.clear()
        
        if self._graph is not None:
            num_vars = self._graph.num_variables()
            for i in range(num_vars):
                meta = self._graph.get_variable(i)
                name = meta.name
                self._node_names.append(name)
                self._name_to_id[name] = meta.id
                self._id_to_name[meta.id] = name
                self._node_states[name] = list(meta.states)

    def _compile(self) -> None:
        # Keep the compiled junction tree alive for the entire engine lifetime.
        # BatchExecutionEngine stores a reference internally.
        self._jt = JunctionTreeCompiler.compile(self._graph, "min_fill")
        self._engine = BatchExecutionEngine(self._jt, self._num_threads, self._chunk_size)
        self._is_compiled = True
        
    def nodes(self) -> List[str]:
        return [n for n in self._node_names if not n.startswith("__")]

    def add_noisy_max(self, node: str, states: List[str], parents: List[str],
                      link_matrices: Dict[str, np.ndarray], leak_probs: np.ndarray) -> None:
        """
        Synthesizes a Noisy-MAX node using Parent Divorcing to keep exact inference fast O(n).
        Adds hidden variables to the underlying `_graph` instead of an exponential CPT.
        
        Args:
            node: Name of the child node.
            states: Ordered states of the child node (lowest effect/leak first, highest last).
            parents: Ordered list of parent names (must already exist in the graph).
            link_matrices: Dict mapping parent_name -> 2D array of shape (parent_states, child_states)
                           where row i is the probability distribution over child states given parent state i.
            leak_probs: 1D array of shape (child_states) representing the background leak distribution.
        """
        if self._graph is None:
            self._graph = Graph()
            
        n_states = len(states)
        
        parent_cardinalities = []
        for p in parents:
            if p not in self._node_names:
                raise ValueError(f"Parent '{p}' does not exist in the graph.")
            meta = self._graph.get_variable(p)
            parent_cardinalities.append(len(meta.states))
            
        leak_name = f"__{node}_Z_leak"
        self._graph.add_variable(leak_name, states)
        self._cpts[leak_name] = np.array(leak_probs, dtype=np.float64)
        
        last_m = leak_name
        
        for i, p in enumerate(parents):
            z_name = f"__{node}_Z_{p}"
            self._graph.add_variable(z_name, states)
            self._graph.add_edge(p, z_name)
            
            matrix = np.asarray(link_matrices[p], dtype=np.float64)
            if matrix.shape != (parent_cardinalities[i], n_states):
                raise ValueError(f"Link matrix for parent '{p}' has shape {matrix.shape}, expected ({parent_cardinalities[i]}, {n_states})")
            
            self._cpts[z_name] = matrix.flatten()
            
            is_last = (i == len(parents) - 1)
            m_name = node if is_last else f"__{node}_M_{p}"
            
            if is_last:
                self._graph.add_variable(node, states)
            else:
                self._graph.add_variable(m_name, states)
                
            self._graph.add_edge(last_m, m_name)
            self._graph.add_edge(z_name, m_name)
            
            max_cpt = np.zeros((n_states, n_states, n_states), dtype=np.float64)
            for a in range(n_states):
                for b in range(n_states):
                    max_cpt[a, b, max(a, b)] = 1.0
                    
            self._cpts[m_name] = max_cpt.flatten()
            last_m = m_name
            
        for n, cpt in self._cpts.items():
            if n.startswith(f"__{node}_") or n == node:
                self._graph.set_cpt(n, cpt)
                
        self._cache_metadata()
        self._is_compiled = False

    def set_equation(self, node: str, expression: callable, parents: List[str]) -> None:
        """
        Creates a deterministic Functional Node by evaluating a Python callable over the Cartesian
        product of its parent states.
        
        Args:
            node: Name of the target node (must already have states defined).
            expression: A Python function taking N strings and returning a string (the node's state).
                        e.g., `lambda a, b: str(int(a) + int(b))`
            parents: Ordered list of parent names.
        """
        import itertools
        
        if self._graph is None:
            raise RuntimeError("Graph must be initialized before adding equations.")
            
        if node not in self._node_names:
            raise KeyError(f"Target node '{node}' does not exist. Add it to graph first.")
            
        target_states = self._node_states[node]
        parent_states_lists = []
        
        for p in parents:
            if p not in self._node_names:
                raise KeyError(f"Parent '{p}' does not exist.")
            parent_states_lists.append(self._node_states[p])
            
        n_states = len(target_states)
        n_rows = 1
        for lst in parent_states_lists:
            n_rows *= len(lst)
            
        cpt = np.zeros(n_rows * n_states, dtype=np.float64)
        
        for i, parent_combo in enumerate(itertools.product(*parent_states_lists)):
            try:
                res = expression(*parent_combo)
            except Exception as e:
                raise RuntimeError(f"Equation evaluation failed on inputs {parent_combo}: {e}")
                
            if res not in target_states:
                raise ValueError(f"Equation returned '{res}', which is not a valid state for '{node}'. Valid: {target_states}")
                
            res_idx = target_states.index(res)
            cpt[i * n_states + res_idx] = 1.0
            
        self.set_cpt(node, cpt.reshape((n_rows, n_states)))

    def get_outcomes(self, node: str) -> List[str]:
        return self._node_states[node]
        
    def parents(self, node: str) -> Tuple[str, ...]:
        meta = self._graph.get_variable(node)
        parent_ids = self._graph.get_parents(meta.id)
        return tuple(self._id_to_name[pid] for pid in parent_ids)
        
    def children(self, node: str) -> List[str]:
        meta = self._graph.get_variable(node)
        child_ids = self._graph.get_children(meta.id)
        return [self._id_to_name[cid] for cid in child_ids]
        
    def get_cpt_shaped(self, node: str) -> np.ndarray:
        flat = self._cpts[node]
        parents = self.parents(node)
        n_card = len(self._node_states[node])
        
        expected_rows = 1
        for p in parents:
            expected_rows *= len(self._node_states[p])
            
        return flat.copy().reshape((expected_rows, n_card))

    def _expected_row_count(self, node: str) -> int:
        parents = self.parents(node)
        expected_rows = 1
        for p in parents:
            expected_rows *= len(self._node_states[p])
        return int(expected_rows)
        
    def set_cpt(self, node: str, shaped: np.ndarray, validate: bool = False) -> None:
        if shaped.ndim != 2:
            raise ValueError(f"CPT for '{node}' must be 2D matrix.")
            
        if validate:
            sums = np.sum(shaped, axis=-1)
            if not np.allclose(sums, 1.0, atol=1e-8):
                raise ValueError(f"CPT rows for '{node}' must deterministically sum strictly to 1.0.")
            if np.any(shaped < -1e-15) or np.any(shaped > 1 + 1e-15):
                raise ValueError(f"CPT values for '{node}' are outside legitimate [0, 1] bounds.")
                
        flat = shaped.astype(np.float64).ravel(order='C')
        
        # This will natively trap structural memory anomalies in C++
        self._graph.set_cpt(node, flat)
        self._cpts[node] = flat
        if self._engine is not None and hasattr(self._engine, "invalidate_workspace_cache"):
            self._engine.invalidate_workspace_cache()

    def set_cpt_batched(self, node: str, shaped_batched: np.ndarray, validate: bool = False) -> None:
        """
        Set a batched CPT tensor for `node` with shape (rows, node_card, batch_size).
        This is a pybncore-specific fast path used by HCL BN-UQ vectorized execution.
        """
        arr = np.asarray(shaped_batched, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"Batched CPT for '{node}' must be 3D (rows, node_card, batch_size).")

        expected_rows = self._expected_row_count(node)
        n_card = len(self._node_states[node])
        if arr.shape[0] != expected_rows or arr.shape[1] != n_card:
            raise ValueError(
                f"Batched CPT for '{node}' has invalid shape {arr.shape}; "
                f"expected ({expected_rows}, {n_card}, batch_size)."
            )
        if arr.shape[2] <= 0:
            raise ValueError(f"Batched CPT for '{node}' must have batch_size > 0.")

        if validate:
            sums = np.sum(arr, axis=1)
            if not np.allclose(sums, 1.0, atol=1e-8):
                raise ValueError(
                    f"Batched CPT rows for '{node}' must sum to 1.0 for every sample."
                )
            if np.any(arr < -1e-15) or np.any(arr > 1 + 1e-15):
                raise ValueError(f"Batched CPT values for '{node}' are outside [0, 1].")

        flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
        self._graph.set_cpt(node, flat)

        # Keep scalar-shaped mirror for APIs expecting 2D retrieval.
        self._cpts[node] = np.ascontiguousarray(arr[:, :, 0], dtype=np.float64).ravel(order="C")
        if self._engine is not None and hasattr(self._engine, "invalidate_workspace_cache"):
            self._engine.invalidate_workspace_cache()

    def make_evidence_matrix(
        self, evidence: Optional[Dict[str, Union[str, int]]], batch_size: int
    ) -> np.ndarray:
        """
        Build a contiguous evidence matrix of shape (batch_size, num_vars) with -1 for unknown.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        num_vars = len(self._name_to_id)
        ev_array = np.full((int(batch_size), num_vars), -1, dtype=np.int32)
        if not evidence:
            return ev_array
        for node, state in evidence.items():
            node_name = str(node)
            if node_name not in self._name_to_id:
                raise KeyError(f"Unknown node '{node_name}' in evidence.")
            if isinstance(state, int):
                state_idx = int(state)
            else:
                outcomes = self._node_states[node_name]
                if state in outcomes:
                    state_idx = outcomes.index(state)
                else:
                    tgt = "".join(ch for ch in str(state).lower() if ch.isalnum())
                    state_idx = -1
                    for i, out in enumerate(outcomes):
                        norm_out = "".join(ch for ch in str(out).lower() if ch.isalnum())
                        if norm_out == tgt:
                            state_idx = i
                            break
                    if state_idx < 0:
                        raise ValueError(
                            f"Unknown outcome '{state}' for node '{node_name}' "
                            f"(outcomes={outcomes})"
                        )
            if state_idx < 0 or state_idx >= len(self._node_states[node_name]):
                raise IndexError(
                    f"Evidence state index {state_idx} out of range for node '{node_name}'."
                )
            ev_array[:, self._name_to_id[node_name]] = int(state_idx)
        return ev_array

    def set_evidence(self, evidence: Optional[Dict[str, Union[str, int]]]) -> None:
        if not evidence:
            self.clear_evidence()
            return
            
        for node, state in evidence.items():
            if isinstance(state, int):
                state = self._node_states[node][state]
            if state not in self._node_states[node]:
                raise ValueError(f"Unknown outcome '{state}' for parent '{node}' (outcomes={self._node_states[node]})")
            self._evidence[node] = state
            
    def clear_evidence(self) -> None:
        self._evidence.clear()
        
    def set_soft_evidence(self, node: str, likelihoods: Dict[str, float]) -> None:
        """
        Sets soft/virtual evidence on a specific node.
        Soft evidence scales the posterior beliefs by these likelihood factors.
        
        Args:
            node: The name of the target node.
            likelihoods: A dictionary mapping state names to likelihood values.
                         States missing from the dictionary receive 0.0 likelihood.
        """
        if not self._is_compiled:
            self._compile()
        if node not in self._name_to_id:
            raise ValueError(f"Unknown node '{node}'")
        node_id = self._name_to_id[node]
        states = self._node_states[node]
        lvec = np.zeros(len(states), dtype=np.float64)
        for s, v in likelihoods.items():
            if s not in states:
                raise ValueError(f"Unknown outcome '{s}' for node '{node}'")
            lvec[states.index(s)] = float(v)
        self._engine.set_soft_evidence(node_id, lvec)

    def set_soft_evidence_matrix(self, node: str, matrix: np.ndarray) -> None:
        """
        Sets a batched soft evidence matrix for a specific node.
        
        Args:
            node: The name of the target node.
            matrix: 2D numpy array of shape (batch_size, n_states), 
                    acting as per-row likelihood multipliers.
        """
        if not self._is_compiled:
            self._compile()
        if node not in self._name_to_id:
            raise ValueError(f"Unknown node '{node}'")
        arr = np.ascontiguousarray(matrix, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("set_soft_evidence_matrix: must be 2D array.")
        node_id = self._name_to_id[node]
        if arr.shape[1] != len(self._node_states[node]):
            raise ValueError(f"set_soft_evidence_matrix: expected {len(self._node_states[node])} columns for node '{node}'.")
        self._engine.set_soft_evidence_matrix(node_id, arr)

    def clear_soft_evidence(self) -> None:
        """
        Clears all previously set soft evidence and likelihoods.
        """
        if self._engine is not None:
            self._engine.clear_soft_evidence()

    def update_beliefs(self) -> None:
        """
        Forces a calibration sequence. Used heavily by HCL mapping
        validation to catch and cache impossible joint evidence pairs.
        """
        nodes = self.nodes()
        if nodes:
            self.batch_query_marginals([nodes[0]])
        
    def query_p(self, node: str, state: Union[str, int]) -> float:
        marginals = self.batch_query_marginals([node])
        pmf_dict = marginals[node]
        
        if isinstance(state, int):
            state = self._node_states[node][state]
            
        return float(pmf_dict[state])

    def batch_query_marginals(self, nodes: Sequence[str], evidence_matrix: Optional[np.ndarray] = None) -> Union[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
        if not self._is_compiled:
            self._compile()

        if not nodes:
            return {}
            
        is_batch = evidence_matrix is not None
        
        if is_batch:
            ev_array = np.ascontiguousarray(evidence_matrix, dtype=np.int32)
            if ev_array.ndim != 2:
                raise ValueError("evidence_matrix must be a 2D int32 array.")
            if ev_array.shape[1] != len(self._name_to_id):
                raise ValueError(
                    f"evidence_matrix second dimension must equal number of nodes "
                    f"({len(self._name_to_id)})."
                )
            batch_size = int(ev_array.shape[0])
            if batch_size <= 0:
                raise ValueError("evidence_matrix must contain at least one row.")
        else:
            batch_size = 1
            num_vars = len(self._name_to_id)
            ev_array = np.full((1, num_vars), -1, dtype=np.int32)
            
            # Form single matrix from scalar dict tracker map
            for e_node, e_state in self._evidence.items():
                node_id = self._name_to_id[e_node]
                state_idx = self._node_states[e_node].index(e_state)
                ev_array[0, node_id] = state_idx

        query_nodes = [str(n) for n in nodes]
        results: Dict[str, Union[Dict[str, float], np.ndarray]] = {}

        # Fast path: one calibrated inference pass for all requested query vars.
        if hasattr(self._engine, "evaluate_multi"):
            query_ids = np.asarray([self._name_to_id[n] for n in query_nodes], dtype=np.int64)
            offsets = [0]
            for n in query_nodes:
                offsets.append(offsets[-1] + len(self._node_states[n]))
            offset_arr = np.asarray(offsets, dtype=np.int64)
            total_states = int(offset_arr[-1])

            output = np.zeros((batch_size, total_states), dtype=np.float64, order='C')
            self._engine.evaluate_multi(ev_array, output, query_ids, offset_arr)

            for i, query_node in enumerate(query_nodes):
                start = int(offset_arr[i])
                end = int(offset_arr[i + 1])
                block = output[:, start:end]
                if is_batch:
                    results[query_node] = block
                else:
                    pmf = block[0]
                    results[query_node] = {
                        s_label: float(pmf[j])
                        for j, s_label in enumerate(self._node_states[query_node])
                    }
            return results

        # Compatibility fallback: evaluate each query variable separately.
        for query_node in query_nodes:
            node_id = self._name_to_id[query_node]
            n_states = len(self._node_states[query_node])
            output = np.zeros((batch_size, n_states), dtype=np.float64, order='C')
            self._engine.evaluate(ev_array, output, node_id)
            if is_batch:
                results[query_node] = output
            else:
                pmf = output[0]
                results[query_node] = {
                    s_label: float(pmf[i])
                    for i, s_label in enumerate(self._node_states[query_node])
                }

        return results

    def batch_query_map(self, evidence_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return MAP state indices with shape (batch_size, num_nodes).
        """
        if not self._is_compiled:
            self._compile()

        if evidence_matrix is None:
            ev_array = self.make_evidence_matrix(self._evidence, batch_size=1)
        else:
            ev_array = np.ascontiguousarray(evidence_matrix, dtype=np.int32)
            if ev_array.ndim != 2:
                raise ValueError("evidence_matrix must be a 2D int32 array.")
            if ev_array.shape[1] != len(self._name_to_id):
                raise ValueError(
                    f"evidence_matrix second dimension must equal number of nodes "
                    f"({len(self._name_to_id)})."
                )
            if ev_array.shape[0] <= 0:
                raise ValueError("evidence_matrix must contain at least one row.")

        batch_size = int(ev_array.shape[0])
        num_nodes = len(self._name_to_id)
        output = np.empty((batch_size, num_nodes), dtype=np.int32, order="C")

        if hasattr(self._engine, "evaluate_map"):
            self._engine.evaluate_map(ev_array, output)
            return output

        # Compatibility fallback for older native modules: marginal argmax per node.
        query_nodes = self.nodes()
        for b in range(batch_size):
            row = ev_array[b : b + 1, :]
            marginals = self.batch_query_marginals(query_nodes, evidence_matrix=row)
            for node in query_nodes:
                probs = np.asarray(marginals[node], dtype=np.float64)[0]
                output[b, self._name_to_id[node]] = int(np.argmax(probs))
        return output

    def query_map(
        self, evidence: Optional[Dict[str, Union[str, int]]] = None
    ) -> Dict[str, str]:
        """
        Return MAP assignment as {node_name: state_name}.
        """
        if evidence is None:
            map_idx = self.batch_query_map()[0]
        else:
            ev_array = self.make_evidence_matrix(evidence, batch_size=1)
            map_idx = self.batch_query_map(ev_array)[0]

        result: Dict[str, str] = {}
        for node in self._node_names:
            idx = int(map_idx[self._name_to_id[node]])
            outcomes = self._node_states[node]
            if idx < 0 or idx >= len(outcomes):
                raise RuntimeError(
                    f"Native MAP result produced invalid state index {idx} for node '{node}'."
                )
            result[node] = outcomes[idx]
        return result

    def sensitivity(
        self,
        query_node: str,
        query_state: str,
        target_node: str,
        parent_config: Tuple[str, ...],
        target_state: str,
        sweep_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Computes the sensitivity of P(query_node=query_state | E) 
        with respect to the parameter P(target_node=target_state | parent_config).
        
        Args:
            query_node: The output node to observe.
            query_state: The state of the output node.
            target_node: The node whose CPT parameter varies.
            parent_config: A tuple of state names for the target node's parents.
            target_state: The specific state of the target node whose probability varies.
            sweep_range: 1D numpy array of probabilities (0.0 to 1.0) to test.
            
        Returns:
            Dictionary with keys:
                "theta": the tested parameter values
                "posterior": the corresponding P(query_node=query_state | E) values
        """
        if not self._is_compiled:
            self._compile()
            
        if query_node not in self._name_to_id:
            raise KeyError(f"Unknown query node '{query_node}'")
        if target_node not in self._name_to_id:
            raise KeyError(f"Unknown target node '{target_node}'")
            
        parents = self.parents(target_node)
        if len(parent_config) != len(parents):
            raise ValueError(f"parent_config length ({len(parent_config)}) does not match parents count ({len(parents)}) for {target_node}.")
            
        cpt_shaped = self.get_cpt_shaped(target_node).copy()
        
        row_idx = 0
        stride = 1
        for i in range(len(parents) - 1, -1, -1):
            p = parents[i]
            p_state = parent_config[i]
            if p_state not in self._node_states[p]:
                raise ValueError(f"Unknown state '{p_state}' for parent '{p}'")
            s_idx = self._node_states[p].index(p_state)
            row_idx += s_idx * stride
            stride *= len(self._node_states[p])
            
        outcomes = self._node_states[target_node]
        if target_state not in outcomes:
            raise ValueError(f"Unknown target state '{target_state}' for node '{target_node}'")
        target_state_idx = outcomes.index(target_state)
        
        orig_row = cpt_shaped[row_idx, :].copy()
        orig_val = orig_row[target_state_idx]
        
        n_points = len(sweep_range)
        results = np.zeros(n_points, dtype=np.float64)
        
        # Suppress validation on self.set_cpt inside the tight loop since standard floating drift is fine
        for i, val in enumerate(sweep_range):
            new_row = orig_row.copy()
            new_row[target_state_idx] = float(val)
            
            rem_orig = 1.0 - orig_val
            rem_new = 1.0 - val
            
            mask = np.ones(len(new_row), dtype=bool)
            mask[target_state_idx] = False
            
            if rem_orig > 1e-12:
                new_row[mask] = orig_row[mask] * (rem_new / rem_orig)
            else:
                n_others = np.sum(mask)
                if n_others > 0:
                    new_row[mask] = rem_new / n_others
                    
            cpt_shaped[row_idx, :] = new_row
            self.set_cpt(target_node, cpt_shaped, validate=False)
            
            marginals = self.batch_query_marginals([query_node])
            if query_node in marginals:
                m = marginals[query_node]
                if isinstance(m, dict):
                    results[i] = m[query_state]
                else:
                    s_idx = self._node_states[query_node].index(query_state)
                    results[i] = m[0, s_idx]
            
        # Restore original CPT
        cpt_shaped[row_idx, :] = orig_row
        self.set_cpt(target_node, cpt_shaped, validate=False)
        
        return {"theta": np.asarray(sweep_range), "posterior": results}

    def sensitivity_ranking(
        self, query_node: str, query_state: str, n_top: int = 10, epsilon: float = 0.05
    ) -> List[Tuple[str, Tuple[str, ...], str, float]]:
        """
        Ranks all CPT parameters in the network by their local derivative
        effect on P(query_node=query_state | E).
        
        Returns:
            List of (target_node, parent_config, target_state, sensitivity_score)
        """
        if not self._is_compiled:
            self._compile()
            
        orig_marginals = self.batch_query_marginals([query_node])
        m = orig_marginals[query_node]
        if isinstance(m, dict):
            orig_p = m[query_state]
        else:
            s_idx = self._node_states[query_node].index(query_state)
            orig_p = m[0, s_idx]
            
        rankings = []
        
        for node in self._node_names:
            cpt_shaped = self.get_cpt_shaped(node).copy()
            parents = self.parents(node)
            states = self._node_states[node]
            rows, card = cpt_shaped.shape
            
            for r in range(rows):
                p_config = []
                rem = r
                stride = rows
                for p in parents:
                    scard = len(self._node_states[p])
                    stride //= scard
                    s_idx = rem // stride
                    rem %= stride
                    p_config.append(self._node_states[p][s_idx])
                p_config_tuple = tuple(p_config)
                
                for c in range(card):
                    orig_val = cpt_shaped[r, c]
                    if orig_val + epsilon <= 1.0:
                        test_val = orig_val + epsilon
                        delta_theta = epsilon
                    else:
                        test_val = orig_val - epsilon
                        delta_theta = -epsilon
                        
                    if test_val < 0.0:
                        continue 
                        
                    new_row = cpt_shaped[r, :].copy()
                    new_row[c] = test_val
                    rem_orig = 1.0 - orig_val
                    rem_new = 1.0 - test_val
                    
                    mask = np.ones(card, dtype=bool)
                    mask[c] = False
                    
                    if rem_orig > 1e-12:
                        new_row[mask] = cpt_shaped[r, mask] * (rem_new / rem_orig)
                    else:
                        n_others = np.sum(mask)
                        if n_others > 0:
                            new_row[mask] = rem_new / n_others
                            
                    test_cpt = cpt_shaped.copy()
                    test_cpt[r, :] = new_row
                    self.set_cpt(node, test_cpt, validate=False)
                    
                    new_marginals = self.batch_query_marginals([query_node])
                    m_new = new_marginals[query_node]
                    if isinstance(m_new, dict):
                        new_p = m_new[query_state]
                    else:
                        q_idx = self._node_states[query_node].index(query_state)
                        new_p = m_new[0, q_idx]
                    
                    derivative = abs((new_p - orig_p) / delta_theta)
                    rankings.append((node, p_config_tuple, states[c], float(derivative)))
                    
            self.set_cpt(node, cpt_shaped, validate=False)
            
        rankings.sort(key=lambda x: x[3], reverse=True)
        return rankings[:n_top]

    def value_of_information(self, query_node: str, candidate_nodes: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Computes the Value of Information (VoI) of observing each candidate node
        with respect to the query_node, given the current evidence.
        VoI is defined as the expected reduction in entropy:
            VoI(V) = H(Q) - sum_s P(V=s) * H(Q | V=s)
            
        Returns:
            List of (candidate_node, voi_score), sorted descending by VoI.
        """
        if not self._is_compiled:
            self._compile()
            
        if query_node not in self._name_to_id:
            raise KeyError(f"Unknown query node '{query_node}'")
            
        if candidate_nodes is None:
            candidate_nodes = [n for n in self._node_names if n != query_node and n not in self._evidence]
            
        nodes_to_query = [query_node] + [v for v in candidate_nodes if v != query_node and v not in self._evidence]
        candidate_nodes = [v for v in candidate_nodes if v not in self._evidence]
        
        if not candidate_nodes:
            return []
            
        priors = self.batch_query_marginals(nodes_to_query)
        q_prior = priors[query_node]
        if isinstance(q_prior, dict):
            q_probs = np.array([q_prior[s] for s in self._node_states[query_node]])
        else:
            q_probs = q_prior[0]
            
        q_probs_nz = q_probs[q_probs > 1e-15]
        h_q = -np.sum(q_probs_nz * np.log2(q_probs_nz))
        
        if h_q < 1e-12:
            return [(v, 0.0) for v in candidate_nodes]
            
        row_descriptors = []
        total_rows = sum(len(self._node_states[v]) for v in candidate_nodes)
        
        ev_matrix = self.make_evidence_matrix(self._evidence, batch_size=total_rows)
        
        row = 0
        for v in candidate_nodes:
            v_prior_dict = priors[v]
            if isinstance(v_prior_dict, dict):
                v_probs = np.array([v_prior_dict[s] for s in self._node_states[v]])
            else:
                v_probs = v_prior_dict[0]
                
            v_id = self._name_to_id[v]
            for s_idx in range(len(self._node_states[v])):
                ev_matrix[row, v_id] = s_idx
                row_descriptors.append((v, v_probs[s_idx]))
                row += 1
                
        conditional_marginals = self.batch_query_marginals([query_node], evidence_matrix=ev_matrix)
        q_cond_matrix = conditional_marginals[query_node]
        
        expected_h = {v: 0.0 for v in candidate_nodes}
        for r in range(total_rows):
            v, p_v_s = row_descriptors[r]
            if p_v_s > 1e-15:
                if isinstance(q_cond_matrix, np.ndarray):
                    q_c_probs = q_cond_matrix[r, :]
                else:
                    q_c_probs = np.array([q_cond_matrix[s] for s in self._node_states[query_node]])
                q_c_probs_nz = q_c_probs[q_c_probs > 1e-15]
                h_q_cond = -np.sum(q_c_probs_nz * np.log2(q_c_probs_nz))
                expected_h[v] += p_v_s * h_q_cond
                
        results = []
        for v in candidate_nodes:
            voi = max(0.0, float(h_q - expected_h[v]))
            results.append((v, voi))
                
        results.sort(key=lambda x: x[1], reverse=True)
        return results
