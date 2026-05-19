"""Per-node DMTT state: link reliability, Beta trust, and collaboration scoring.

Implements Algorithms 1 and 4 from the DMTT paper.

Notation mapping (paper → code):
    ĉ_ij^t          → _c_hat[j]
    α_ij / β_ij     → _alpha[j] / _beta[j]   (Beta distribution parameters)
    R_ij            → posterior mean  α/(α+β)
    U_ij            → posterior std   sqrt(αβ / ((α+β)² (α+β+1)))
    T_ij^{topo,t}   → topo_trust(j)           (Alg. 4 output)
    s_ij^{model,t}  → computed externally, passed into collab_score()
    q_ij^t          → collab_score()
    C_i^t           → top_b()
"""

import math
from typing import Dict, List, Optional

from murmura.config.schema import DMTTConfig


class DMTTNodeState:
    """Tracks DMTT state for all neighbours of a single node i.

    Thread-safety: not thread-safe; use from a single process only.
    """

    def __init__(self, node_id: int, cfg: DMTTConfig):
        self.node_id = node_id
        self.cfg     = cfg

        # ĉ_ij ∈ [0,1] — EMA of received acks
        self._c_hat: Dict[int, float] = {}
        # Beta distribution params — positive / negative trust evidence
        self._alpha: Dict[int, float] = {}
        self._beta:  Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _init(self, j: int) -> None:
        if j not in self._c_hat:
            self._c_hat[j] = 0.5   # neutral initial link reliability
        if j not in self._alpha:
            self._alpha[j] = 1.0   # uniform Beta(1,1) prior
            self._beta[j]  = 1.0

    # ------------------------------------------------------------------
    # Algorithm 1 — link reliability update
    # ------------------------------------------------------------------

    def update_link_reliability(self, j: int, received: bool) -> None:
        """ĉ_ij^{t+1} = (1-ρ)ĉ_ij^t + ρ · ack_ij^t"""
        self._init(j)
        rho = self.cfg.rho
        self._c_hat[j] = (1.0 - rho) * self._c_hat[j] + rho * (1.0 if received else 0.0)

    # ------------------------------------------------------------------
    # Algorithm 4 — source trust update
    # ------------------------------------------------------------------

    def update_trust(
        self,
        j: int,
        d: float,          # direct confirmations  (positive evidence)
        x: float,          # contradictions        (negative evidence)
        c: float = 0.0,    # corroboration         (positive evidence)
    ) -> None:
        """α_ij^{t+1} = λα + w_d·d + w_c·c;   β_ij^{t+1} = λβ + w_x·x"""
        self._init(j)
        cfg = self.cfg
        self._alpha[j] = cfg.lambda_forget * self._alpha[j] + cfg.w_d * d + cfg.w_c * c
        self._beta[j]  = cfg.lambda_forget * self._beta[j]  + cfg.w_x * x
        self._alpha[j] = max(0.01, self._alpha[j])
        self._beta[j]  = max(0.01, self._beta[j])

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def topo_trust(self, j: int) -> float:
        """T_ij^topo = R_ij · exp(-η · max(0, U_ij - τ_U))

        R_ij = α/(α+β)  (posterior mean)
        U_ij = sqrt(αβ / ((α+β)²(α+β+1)))  (posterior std / uncertainty)
        """
        self._init(j)
        a, b = self._alpha[j], self._beta[j]
        s    = a + b
        R    = a / s
        U    = math.sqrt(max(0.0, a * b / (s * s * (s + 1.0))))
        cfg  = self.cfg
        return R * math.exp(-cfg.eta * max(0.0, U - cfg.tau_U))

    def link_reliability(self, j: int) -> float:
        self._init(j)
        return self._c_hat[j]

    def model_score(self, a_ij: float, u_bar_ij: float = 0.0) -> float:
        """s_ij^{model} with optional uncertainty penalty.

        s_base = (1 - ū) · (w_a · a_ij + (1 - w_a))
        s      = s_base · exp(-(ū - τ_u))  if ū > τ_u, else s_base
        """
        cfg    = self.cfg
        s_base = (1.0 - u_bar_ij) * (cfg.w_a * a_ij + (1.0 - cfg.w_a))
        if u_bar_ij > cfg.tau_u:
            s_base = s_base * math.exp(-(u_bar_ij - cfg.tau_u))
        return max(0.0, s_base)

    def collab_score(
        self,
        j: int,
        s_model: float,
        c_comm: float = 0.0,
    ) -> float:
        """q_ij = λ1·s_model + λ2·T_topo + λ3·ĉ - λ4·c_comm"""
        cfg = self.cfg
        T   = self.topo_trust(j)
        c   = self.link_reliability(j)
        return cfg.lambda1 * s_model + cfg.lambda2 * T + cfg.lambda3 * c - cfg.lambda4 * c_comm

    # ------------------------------------------------------------------
    # TopB selection (Algorithm 1, line: C_i^t ← TopB_j q_ij^t)
    # ------------------------------------------------------------------

    def top_b(
        self,
        candidates: List[int],
        model_scores: Dict[int, float],
        B: int,
    ) -> List[int]:
        """Return up to B candidates ranked by collaboration score (descending)."""
        if not candidates:
            return []
        scored = [
            (j, self.collab_score(j, model_scores.get(j, 0.5)))
            for j in candidates
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [j for j, _ in scored[:B]]

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def state_summary(self, peers: Optional[List[int]] = None) -> Dict[int, dict]:
        """Return a summary dict for logging / debugging."""
        targets = peers if peers is not None else list(self._alpha.keys())
        return {
            j: {
                "c_hat":      self.link_reliability(j),
                "T_topo":     self.topo_trust(j),
                "alpha":      self._alpha.get(j, 1.0),
                "beta":       self._beta.get(j, 1.0),
            }
            for j in targets
        }
