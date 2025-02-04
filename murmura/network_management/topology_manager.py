from typing import Dict, List

from murmura.network_management.topology import TopologyConfig, TopologyType


class TopologyManager:
    """
    Generates and manages client connection topologies
    """

    def __init__(self, num_clients: int, config: TopologyConfig):
        self.num_clients = num_clients
        self.config = config
        self.adjacency_list = self._generate_topology()

    def _generate_topology(self) -> Dict[int, List[int]]:
        """
        Generate adjacency list based on topology type

        :return: Adjacency list of neighbours
        """
        generators = {
            TopologyType.STAR: self._star,
            TopologyType.RING: self._ring,
            TopologyType.COMPLETE: self._complete,
            TopologyType.LINE: self._line,
            TopologyType.CUSTOM: self._custom,
        }

        return generators[self.config.topology_type]()

    def _star(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for star topology

        :return: Adjacency list of neighbours
        """
        hub = self.config.hub_index % self.num_clients
        return {
            i: [hub] if i != hub else [n for n in range(self.num_clients) if n != hub]
            for i in range(self.num_clients)
        }

    def _ring(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for ring topology

        :return: Adjacency list of neighbours
        """
        return {
            i: [(i - 1) % self.num_clients, (i + 1) % self.num_clients]
            for i in range(self.num_clients)
        }

    def _complete(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for complete topology

        :return: Adjacency list of neighbours
        """
        return {
            i: [n for n in range(self.num_clients) if n != i]
            for i in range(self.num_clients)
        }

    def _line(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for line topology

        :return: Adjacency list of neighbours
        """
        return {
            i: [i - 1] if i > 0 else [] + [i + 1] if i < self.num_clients - 1 else []
            for i in range(self.num_clients)
        }

    def _custom(self) -> Dict[int, List[int]]:
        """
        Returns adjacency list for custom topology

        :return: Adjacency list of neighbours
        """
        if not self.config.adjacency_list:
            raise ValueError("Custom topology requires adjacency list")
        return self.config.adjacency_list
