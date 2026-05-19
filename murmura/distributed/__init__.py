"""ZeroMQ-based distributed backend for Murmura.

Transforms Murmura from a single-process simulation into a framework where each
node runs as an independent OS process communicating via ZeroMQ sockets.  Round
timing is governed by the wall clock — no coordinator sends round-start signals.

Single-machine usage (all nodes on one host):
    runner = DistributedRunner(config_path)
    history = runner.run(verbose=True)

Multi-machine usage (one node per host):
    # Head node — starts monitor + node 0:
    #   murmura run config.yaml      (with backend: distributed)
    # Each worker node:
    #   murmura run-node config.yaml --node-id 1 --t-start <epoch_float>
"""

from murmura.distributed.runner import DistributedRunner

__all__ = ["DistributedRunner"]
