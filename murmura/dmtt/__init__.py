"""DMTT — Dynamic MURMURA with Trusted Topology.

Implements the four algorithms from the DMTT paper:
  1. Main loop with link-reliability EMA and TopB collaborator selection
  2. Topology-claim processing and edge-confidence accumulation
  3. Controlled forwarding (H=1 in the distributed implementation)
  4. Beta-evidence source-trust update

Entry points:
    DMTTNodeState  — per-node state tracker (state.py)
    DMTTNodeProcess — ZMQ node process with DMTT trust (node_process.py)
"""

from murmura.dmtt.state import DMTTNodeState
from murmura.dmtt.node_process import DMTTNodeProcess

__all__ = ["DMTTNodeState", "DMTTNodeProcess"]
