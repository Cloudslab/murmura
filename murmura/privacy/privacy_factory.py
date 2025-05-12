from typing import Optional

from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.privacy.privacy_config import (
    PrivacyConfig,
    PrivacyMode,
    PrivacyMechanismType,
)
from murmura.privacy.privacy_manager import PrivacyManager


class PrivacyFactory:
    """
    Factory for creating privacy manager instances based on configuration.
    """

    @staticmethod
    def create(
        config: PrivacyConfig, topology_config: Optional[TopologyConfig] = None
    ) -> PrivacyManager:
        """
        Create a PrivacyManager instance based on the provided configuration.

        Args:
            config (PrivacyConfig): Configuration object containing privacy settings.
            topology_config (Optional[TopologyConfig]): Topology configuration for the network.

        Returns:
            PrivacyManager: An instance of PrivacyManager configured with the provided settings.
        """
        privacy_manager = PrivacyManager(config)

        if topology_config and config.enabled:
            topology_type = topology_config.topology_type
            privacy_mode = config.privacy_mode

            if not privacy_manager.is_compatible_with_topology(topology_type):
                if privacy_mode == PrivacyMode.CENTRAL:
                    raise ValueError(
                        f"Central Differential Privacy is not compatible with topology type: {topology_type.value}. "
                        f"Central DP requires a centralized topology like STAR or COMPLETE."
                    )

                compatible_modes = []
                if topology_type in [
                    TopologyConfig.STAR,
                    TopologyConfig.COMPLETE,
                ]:
                    compatible_modes = ["LOCAL", "CENTRAL"]
                else:
                    compatible_modes = ["LOCAL"]

                raise ValueError(
                    f"Privacy mode {privacy_mode.value} is not compatible with topology type: {topology_type.value}. "
                    f"Compatible modes are: {', '.join(compatible_modes)}."
                )

        return privacy_manager

    @staticmethod
    def create_compatible_privacy_config(
        topology_type: TopologyType, desired_mode: Optional[PrivacyMode] = None
    ) -> PrivacyConfig:
        """
        Create a compatible privacy configuration based on the topology type and desired privacy mode.

        Args:
            topology_type (TopologyType): The topology type to check compatibility with.
            desired_mode (Optional[PrivacyMode]): The desired privacy mode.

        Returns:
            PrivacyConfig: A compatible privacy configuration.
        """
        if desired_mode == PrivacyMode.CENTRAL:
            if topology_type not in [TopologyType.STAR, TopologyType.COMPLETE]:
                raise ValueError(
                    f"Central Differential Privacy is not compatible with topology type: {topology_type.value}. "
                    f"Central DP requires a centralized topology like STAR or COMPLETE."
                )

            return PrivacyConfig(
                enabled=True,
                privacy_mode=PrivacyMode.CENTRAL,
                mechanism_type=PrivacyMechanismType.GAUSSIAN,
            )

        return PrivacyConfig(
            enabled=True,
            privacy_mode=PrivacyMode.LOCAL,
            mechanism_type=PrivacyMechanismType.GAUSSIAN,
        )
