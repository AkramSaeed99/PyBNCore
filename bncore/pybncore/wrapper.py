import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ._core import Graph, JunctionTreeCompiler, BatchExecutionEngine, VariableMetadata
from .io import read_xdsl

class PyBNCoreWrapper:
    """
    HCL-compatible wrapper integrating PyBNCore natively into legacy BDD_Engine ecosystems.
    Matches exact API specifications of SMILE backends (bn_smile.py / bn_pgmpy.py).
    """
    def __init__(self, model_path: Optional[str] = None):
        self._graph: Optional[Graph] = None
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
        self._node_names = list(self._cpts.keys())
        self._name_to_id.clear()
        self._id_to_name.clear()
        self._node_states.clear()
        
        for name in self._node_names:
            meta = self._graph.get_variable(name)
            self._name_to_id[name] = meta.id
            self._id_to_name[meta.id] = name
            self._node_states[name] = list(meta.states)

    def _compile(self) -> None:
        jt = JunctionTreeCompiler.compile(self._graph, "min_fill")
        self._engine = BatchExecutionEngine(jt, self._num_threads, self._chunk_size)
        self._is_compiled = True
        
    def nodes(self) -> List[str]:
        return self._node_names.copy()

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
        
        # When graph parametrics change natively in BDD environments without structure updates, the engine needs resync:
        self._compile()

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
        
    def query_p(self, node: str, state: Union[str, int]) -> float:
        marginals = self.batch_query_marginals([node])
        pmf_dict = marginals[node]
        
        if isinstance(state, int):
            state = self._node_states[node][state]
            
        return float(pmf_dict[state])

    def batch_query_marginals(self, nodes: Sequence[str], evidence_matrix: Optional[np.ndarray] = None) -> Union[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
        if not self._is_compiled:
            self._compile()
            
        is_batch = evidence_matrix is not None
        
        if is_batch:
            batch_size = evidence_matrix.shape[0]
            ev_array = np.ascontiguousarray(evidence_matrix, dtype=np.int32)
        else:
            batch_size = 1
            num_vars = len(self._name_to_id)
            ev_array = np.full((1, num_vars), -1, dtype=np.int32)
            
            # Form single matrix from scalar dict tracker map
            for e_node, e_state in self._evidence.items():
                node_id = self._name_to_id[e_node]
                state_idx = self._node_states[e_node].index(e_state)
                ev_array[0, node_id] = state_idx
                
        results = {}
        for query_node in nodes:
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
