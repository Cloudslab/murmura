"""Launches all distributed Murmura processes on a single machine.

For multi-machine deployments, start the monitor process on any host with
network access, then start each node with the CLI's `run-node` command on its
own host.  This module handles the single-machine case where all processes
share the same filesystem (IPC).

Process topology:
    monitor process  — 1 process, passive PULL-only metrics collector
    node processes   — N processes (one per node), each runs NodeProcess

Round timing is governed by the system wall clock.  The runner computes a
shared t_start = now + startup_grace_s so all nodes begin round 0 at the
same absolute time, independent of any coordinator signal.
"""

import multiprocessing as mp
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Set

from murmura.config.loader import load_config
from murmura.config.schema import Config
from murmura.distributed.endpoints import Endpoints
from murmura.utils.factories import build_attack


# ---------------------------------------------------------------------------
# Subprocess entry points (module-level so they are picklable with 'spawn')
# ---------------------------------------------------------------------------

def _monitor_main(
    num_nodes: int,
    dist_cfg_dict: dict,
    run_id: str,
    rounds: int,
    t_start: float,
    compromised_nodes: Set[int],
    verbose: bool,
    result_queue: mp.Queue,
) -> None:
    """Entry point for the monitor subprocess."""
    try:
        from murmura.config.schema import DistributedConfig
        from murmura.distributed.endpoints import Endpoints
        from murmura.distributed.monitor import Monitor

        dist_cfg  = DistributedConfig(**dist_cfg_dict)
        endpoints = Endpoints(dist_cfg, num_nodes, run_id)
        monitor   = Monitor(
            num_nodes=num_nodes,
            endpoints=endpoints,
            rounds=rounds,
            t_start=t_start,
            round_duration_s=dist_cfg.round_duration_s,
            compromised_nodes=compromised_nodes,
            verbose=verbose,
        )
        history = monitor.run()
        result_queue.put(("ok", history))
    except Exception:
        import traceback
        result_queue.put(("error", traceback.format_exc()))


def _node_main(
    node_id: int,
    config_path: str,
    dist_cfg_dict: dict,
    num_nodes: int,
    run_id: str,
    t_start: float,
) -> None:
    """Entry point for a node subprocess.

    Uses DMTTNodeProcess when config.dmtt is set, otherwise NodeProcess.
    """
    try:
        from murmura.config.loader import load_config
        from murmura.config.schema import DistributedConfig
        from murmura.distributed.endpoints import Endpoints

        dist_cfg  = DistributedConfig(**dist_cfg_dict)
        endpoints = Endpoints(dist_cfg, num_nodes, run_id)

        config = load_config(config_path)
        if config.dmtt is not None:
            from murmura.dmtt.node_process import DMTTNodeProcess
            proc = DMTTNodeProcess.from_config_path(
                node_id=node_id,
                config_path=config_path,
                endpoints=endpoints,
                t_start=t_start,
            )
        else:
            from murmura.distributed.node_process import NodeProcess
            proc = NodeProcess.from_config_path(
                node_id=node_id,
                config_path=config_path,
                endpoints=endpoints,
                t_start=t_start,
            )
        proc.run()
    except Exception:
        import traceback
        print(f"[Node {node_id}] FATAL:\n{traceback.format_exc()}", flush=True)


# ---------------------------------------------------------------------------
# Public runner class
# ---------------------------------------------------------------------------

class DistributedRunner:
    """Manages all processes for a single-machine distributed Murmura run.

    Args:
        config_path: Path to the experiment YAML/JSON config file.
    """

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config: Config = load_config(self.config_path)

    def run(self, verbose: bool = False) -> Dict[str, List[Any]]:
        """Launch monitor + node processes and block until training completes.

        Returns:
            Training history dict identical in structure to Network.train().
        """
        cfg       = self.config
        num_nodes = cfg.topology.num_nodes
        run_id    = uuid.uuid4().hex[:8]

        attack      = build_attack(cfg)
        compromised: Set[int] = attack.get_compromised_nodes() if attack else set()

        endpoints = Endpoints(cfg.distributed, num_nodes, run_id)
        endpoints.ensure_dirs()

        dist_cfg_dict = cfg.distributed.model_dump()

        # t_start is a monotonic timestamp.  All processes on the same machine
        # share the same clock so monotonic is safe here.  For multi-machine
        # runs the runner hands t_start to nodes via command-line args instead.
        t_start = time.monotonic() + cfg.distributed.startup_grace_s

        # Print run_id + t_start so multi-machine operators can start workers.
        print(
            f"[DistributedRunner] run_id={run_id}  t_start={t_start:.3f}  "
            f"(startup_grace={cfg.distributed.startup_grace_s}s)",
            flush=True,
        )

        result_queue: mp.Queue = mp.Queue()

        # Start monitor first so its PULL socket is bound before nodes connect.
        monitor_proc = mp.Process(
            target=_monitor_main,
            args=(
                num_nodes,
                dist_cfg_dict,
                run_id,
                cfg.experiment.rounds,
                t_start,
                compromised,
                verbose,
                result_queue,
            ),
            name="murmura-monitor",
            daemon=True,
        )
        monitor_proc.start()

        # Brief pause to let the monitor bind before nodes attempt to connect.
        time.sleep(0.2)

        # Start one node process per node.
        node_procs: List[mp.Process] = []
        for node_id in range(num_nodes):
            p = mp.Process(
                target=_node_main,
                args=(
                    node_id,
                    str(self.config_path),
                    dist_cfg_dict,
                    num_nodes,
                    run_id,
                    t_start,
                ),
                name=f"murmura-node-{node_id}",
                daemon=True,
            )
            p.start()
            node_procs.append(p)

        # Block until monitor finishes (all rounds collected or deadline).
        monitor_proc.join()

        # Give nodes a moment to finish their last metrics push, then clean up.
        for p in node_procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        if result_queue.empty():
            raise RuntimeError("Monitor exited without producing a result.")

        status, value = result_queue.get_nowait()
        if status == "error":
            raise RuntimeError(f"Monitor failed:\n{value}")

        return value
