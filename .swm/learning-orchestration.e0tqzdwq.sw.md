---
title: Learning Orchestration
---
# Introduction

This document will walk you through the "Learning Orchestration" implementation. The purpose of this implementation is to manage a distributed learning environment using Ray, where virtual client actors participate in federated learning.

We will cover:

1. How virtual client actors are initialized and managed.
2. How data is distributed among these actors.
3. How the topology of the network is applied and managed.
4. How configuration is handled for the orchestration process.

# Virtual client actor initialization

<SwmSnippet path="/murmura/node/client_actor.py" line="10">

---

The <SwmToken path="/murmura/node/client_actor.py" pos="7:2:2" line-data="class VirtualClientActor:">`VirtualClientActor`</SwmToken> class represents a virtual client in the federated learning setup. It is a Ray remote actor, which allows it to run in a distributed manner.

```
    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.data_partition: Optional[List[int]] = None
        self.metadata: Dict[str, Any] = {}
        self.neighbours: List[Any] = []
```

---

</SwmSnippet>

The constructor initializes the client with an ID and prepares it to store data partitions, metadata, and neighboring client references. This setup is crucial for each client to manage its own data and interact with others.

# Data reception and information retrieval

<SwmSnippet path="/murmura/node/client_actor.py" line="16">

---

The <SwmToken path="/murmura/node/client_actor.py" pos="16:3:3" line-data="    def receive_data(">`receive_data`</SwmToken> method allows a client actor to receive a data partition and optional metadata. This is essential for distributing data across clients in a federated learning setup.

```
    def receive_data(
        self, data_partition: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Receive a data partition and metadata dictionary.
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/node/client_actor.py" line="29">

---

Once data is received, the client can provide information about its stored data through the <SwmToken path="/murmura/node/client_actor.py" pos="29:3:3" line-data="    def get_data_info(self) -&gt; Dict[str, Any]:">`get_data_info`</SwmToken> method. This method returns details such as the client ID, data size, and metadata.

```
    def get_data_info(self) -> Dict[str, Any]:
        """
        Return Information about stored data partition.
        :return:
        """
        return {
            "client_id": self.client_id,
            "data_size": len(self.data_partition) if self.data_partition else 0,
            "metadata": self.metadata,
        }
```

---

</SwmSnippet>

# Neighbour management

<SwmSnippet path="/murmura/node/client_actor.py" line="40">

---

The <SwmToken path="/murmura/node/client_actor.py" pos="40:3:3" line-data="    def set_neighbours(self, neighbours: List[Any]) -&gt; None:">`set_neighbours`</SwmToken> method is used to establish communication links between client actors by setting their neighbors.

```
    def set_neighbours(self, neighbours: List[Any]) -> None:
        """
        Set neighbour actors for communication
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/node/client_actor.py" line="48">

---

The <SwmToken path="/murmura/node/client_actor.py" pos="48:3:3" line-data="    def get_neighbours(self) -&gt; List[str]:">`get_neighbours`</SwmToken> method retrieves the <SwmToken path="/murmura/node/client_actor.py" pos="50:3:3" line-data="        Get IDs of neighbouring clients">`IDs`</SwmToken> of neighboring clients, facilitating interaction between them.

```
    def get_neighbours(self) -> List[str]:
        """
        Get IDs of neighbouring clients
```

---

</SwmSnippet>

# Cluster management

<SwmSnippet path="/murmura/orchestration/cluster_manager.py" line="10">

---

The <SwmToken path="/murmura/orchestration/cluster_manager.py" pos="10:2:2" line-data="class ClusterManager:">`ClusterManager`</SwmToken> class is responsible for managing the Ray cluster and the virtual client actors. It initializes the Ray environment if not already done.

```
class ClusterManager:
    """
    Manages Ray cluster and virtual client actors
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.actors: List[Any] = []
        self.topology_manager: Optional[TopologyManager] = None
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/orchestration/cluster_manager.py" line="25">

---

The <SwmToken path="/murmura/orchestration/cluster_manager.py" pos="25:3:3" line-data="    def create_actors(self, num_actors: int, topology: TopologyConfig) -&gt; List[Any]:">`create_actors`</SwmToken> method creates a specified number of virtual client actors and applies the network topology to them.

```
    def create_actors(self, num_actors: int, topology: TopologyConfig) -> List[Any]:
        """
        Create pool of virtual client actors
```

---

</SwmSnippet>

# Data distribution

<SwmSnippet path="murmura/orchestration/cluster_manager.py" line="41">

---

The <SwmToken path="/murmura/orchestration/cluster_manager.py" pos="41:3:3" line-data="    def distribute_data(">`distribute_data`</SwmToken> method distributes data partitions to the actors in a <SwmToken path="/murmura/orchestration/cluster_manager.py" pos="47:13:15" line-data="        Distribute data partitions to actors in round-robin fashion">`round-robin`</SwmToken> fashion. This ensures that each client receives a portion of the data for processing.

```
    def distribute_data(
        self,
        data_partitions: List[List[int]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Distribute data partitions to actors in round-robin fashion

        :param data_partitions: List of data partitions
        :param metadata: Metadata dict

        :return: List of data receipt acknowledgments
        """
        resolved_metadata = metadata or {}

        results = []
        for idx, actor in enumerate(self.actors):
            partition_idx = idx % len(data_partitions)
            results.append(
                actor.receive_data.remote(
                    data_partitions[partition_idx],
                    {**resolved_metadata, "partition_idx": partition_idx},
                )
            )

        return ray.get(results)
```

---

</SwmSnippet>

# Topology application

<SwmSnippet path="murmura/orchestration/cluster_manager.py" line="68">

---

The <SwmToken path="/murmura/orchestration/cluster_manager.py" pos="68:3:3" line-data="    def _apply_topology(self) -&gt; None:">`_apply_topology`</SwmToken> method sets up neighbor relationships between actors based on the topology configuration. This is crucial for defining how clients interact with each other.

```
    def _apply_topology(self) -> None:
        """
        Set neighbour relationships based on topology config
        """
        if not self.topology_manager:
            return
```

---

</SwmSnippet>

# Configuration management

<SwmSnippet path="/murmura/orchestration/orchestration_config.py" line="1">

---

The <SwmToken path="/murmura/orchestration/orchestration_config.py" pos="7:2:2" line-data="class OrchestrationConfig(BaseModel):">`OrchestrationConfig`</SwmToken> class defines the configuration for the orchestration process, including the number of actors, topology, Ray address, dataset details, and partitioning strategy.

```
from pydantic import BaseModel, Field
from typing import Literal, Optional

from murmura.network_management.topology import TopologyConfig


class OrchestrationConfig(BaseModel):
    """
    Configuration object for learning orchestration
    """
```

---

</SwmSnippet>

This configuration is essential for customizing the learning orchestration to fit different scenarios and datasets.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBbXVybXVyYSUzQSUzQW11cnRhemFocg==" repo-name="murmura"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
