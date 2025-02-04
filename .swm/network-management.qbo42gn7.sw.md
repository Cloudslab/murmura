---
title: Network Management
---
# Introduction

This document will walk you through the network management implementation in our system. The purpose of this implementation is to manage and generate different network topologies for client connections.

We will cover:

1. How the topology configuration is defined and validated.
2. How the topology manager generates the adjacency list based on the configuration.
3. The specific methods used to generate each type of topology.

# Topology configuration

<SwmSnippet path="/murmura/network_management/topology.py" line="15">

---

The topology configuration is defined using a Pydantic model, which ensures that the configuration is both structured and validated. The <SwmToken path="/murmura/network_management/topology.py" pos="15:2:2" line-data="class TopologyConfig(BaseModel):">`TopologyConfig`</SwmToken> class specifies the type of topology and any additional parameters needed for specific topologies, such as the hub index for a star topology.

```
class TopologyConfig(BaseModel):
    """
    Configuration Model for Network Topology
    """

    topology_type: TopologyType = Field(
        default=TopologyType.COMPLETE,
        description="Type of network topology between clients",
    )
    hub_index: int = Field(
        default=0, description="Index of hub node (only for star topology)"
    )
    adjacency_list: Optional[Dict[int, List[int]]] = Field(
        default=None, description="Custom adjacency list for CUSTOM topology"
    )
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/network_management/topology.py" line="31">

---

Validation is crucial to ensure that the configuration is correct before generating the topology. For example, the hub index must be non-negative for a star topology.

```
    @model_validator(mode="after")
    def validate_topology(self) -> "TopologyConfig":
        if self.topology_type == TopologyType.STAR and self.hub_index < 0:
            raise ValueError("Hub index cannot be negative")
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/network_management/topology.py" line="36">

---

For custom topologies, an adjacency list must be provided, and it must not contain negative indices.

```
        if self.topology_type == TopologyType.CUSTOM:
            if not self.adjacency_list:
                raise ValueError("Adjacency list required for custom topology")
            for node, neighbors in self.adjacency_list.items():
                if any(n < 0 for n in neighbors):
                    raise ValueError("Negative node indices not allowed")
```

---

</SwmSnippet>

# Topology manager

<SwmSnippet path="/murmura/network_management/topology_manager.py" line="11">

---

The <SwmToken path="/murmura/network_management/topology_manager.py" pos="6:2:2" line-data="class TopologyManager:">`TopologyManager`</SwmToken> class is responsible for generating and managing the client connection topologies. It initializes with the number of clients and a topology configuration, then generates the appropriate adjacency list.

```
    def __init__(self, num_clients: int, config: TopologyConfig):
        self.num_clients = num_clients
        self.config = config
        self.adjacency_list = self._generate_topology()
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/network_management/topology_manager.py" line="20">

---

The <SwmToken path="/murmura/network_management/topology_manager.py" pos="14:9:9" line-data="        self.adjacency_list = self._generate_topology()">`_generate_topology`</SwmToken> method selects the correct generator function based on the topology type specified in the configuration.

```
        :return: Adjacency list of neighbours
        """
        generators = {
            TopologyType.STAR: self._star,
            TopologyType.RING: self._ring,
            TopologyType.COMPLETE: self._complete,
            TopologyType.LINE: self._line,
            TopologyType.CUSTOM: self._custom,
        }
```

---

</SwmSnippet>

# Topology generation methods

<SwmSnippet path="/murmura/network_management/topology_manager.py" line="30">

---

Each topology type has a dedicated method to generate its adjacency list. For example, the <SwmToken path="/murmura/network_management/topology_manager.py" pos="32:3:3" line-data="    def _star(self) -&gt; Dict[int, List[int]]:">`_star`</SwmToken> method creates a star topology by connecting all nodes to a central hub.

```
        return generators[self.config.topology_type]()

    def _star(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for star topology
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/network_management/topology_manager.py" line="44">

---

The <SwmToken path="/murmura/network_management/topology_manager.py" pos="44:3:3" line-data="    def _ring(self) -&gt; Dict[int, List[int]]:">`_ring`</SwmToken> method connects each node to its immediate neighbors, forming a closed loop.

```
    def _ring(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for ring topology
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/network_management/topology_manager.py" line="55">

---

The <SwmToken path="/murmura/network_management/topology_manager.py" pos="55:3:3" line-data="    def _complete(self) -&gt; Dict[int, List[int]]:">`_complete`</SwmToken> method connects every node to every other node, creating a fully connected network.

```
    def _complete(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for complete topology
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/network_management/topology_manager.py" line="66">

---

The <SwmToken path="/murmura/network_management/topology_manager.py" pos="66:3:3" line-data="    def _line(self) -&gt; Dict[int, List[int]]:">`_line`</SwmToken> method connects nodes in a linear sequence, with each node connected to its immediate predecessor and successor.

```
    def _line(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for line topology
```

---

</SwmSnippet>

<SwmSnippet path="/murmura/network_management/topology_manager.py" line="77">

---

Finally, the <SwmToken path="/murmura/network_management/topology_manager.py" pos="77:3:3" line-data="    def _custom(self) -&gt; Dict[int, List[int]]:">`_custom`</SwmToken> method uses a predefined adjacency list provided in the configuration, allowing for flexible and user-defined topologies.

```
    def _custom(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for custom topology
```

---

</SwmSnippet>

This concludes the walkthrough of the network management implementation, highlighting the design decisions and how each component fits into the overall system.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBbXVybXVyYSUzQSUzQW11cnRhemFocg==" repo-name="murmura"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
