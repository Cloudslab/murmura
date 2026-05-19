"""ZMQ endpoint string management for all processes in a distributed run.

Each unique run gets its own subdirectory (IPC) or port range (TCP) so that
concurrent experiments on the same machine do not collide.

Socket roles:
    monitor_pull(run) — monitor binds; nodes connect (PUSH) to send metrics
    node_pull(i)      — node i binds; neighbours connect (PUSH) to send model states

There is no coordinator PUB socket in the wall-clock design — round
synchronisation is handled by the system clock, not by any message.
"""

import os

from murmura.config.schema import DistributedConfig


class Endpoints:
    """Computes ZMQ endpoint strings for every socket in a distributed run."""

    def __init__(self, dist_cfg: DistributedConfig, num_nodes: int, run_id: str):
        self.cfg      = dist_cfg
        self.num_nodes = num_nodes
        self.run_id   = run_id

    # ------------------------------------------------------------------
    # Monitor endpoints  (passive metrics collector)
    # ------------------------------------------------------------------

    def monitor_pull_bind(self) -> str:
        """Address the monitor binds its PULL socket to."""
        if self.cfg.transport == "ipc":
            return f"ipc://{self.cfg.ipc_dir}/{self.run_id}/monitor_pull"
        return f"tcp://0.0.0.0:{self.cfg.coordinator_pull_port}"

    def monitor_pull_connect(self) -> str:
        """Address nodes connect their PUSH (→ monitor) socket to."""
        if self.cfg.transport == "ipc":
            return f"ipc://{self.cfg.ipc_dir}/{self.run_id}/monitor_pull"
        return f"tcp://{self.cfg.host}:{self.cfg.coordinator_pull_port}"

    # ------------------------------------------------------------------
    # Node endpoints
    # ------------------------------------------------------------------

    def node_pull_bind(self, node_id: int) -> str:
        """Address node_id binds its model-receive PULL socket to."""
        if self.cfg.transport == "ipc":
            return f"ipc://{self.cfg.ipc_dir}/{self.run_id}/node_{node_id}"
        return f"tcp://0.0.0.0:{self.cfg.base_port + node_id}"

    def node_pull_connect(self, node_id: int) -> str:
        """Address other nodes connect their PUSH sockets to when sending to node_id."""
        if self.cfg.transport == "ipc":
            return f"ipc://{self.cfg.ipc_dir}/{self.run_id}/node_{node_id}"
        host = self.cfg.host
        if self.cfg.node_hosts and node_id in self.cfg.node_hosts:
            host = self.cfg.node_hosts[node_id]
        return f"tcp://{host}:{self.cfg.base_port + node_id}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create IPC socket directory if needed (ipc transport only)."""
        if self.cfg.transport == "ipc":
            os.makedirs(f"{self.cfg.ipc_dir}/{self.run_id}", exist_ok=True)
