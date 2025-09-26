import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, rice


class WirelessDataGenerator:
    """
    Generates synthetic wireless network data for resource allocation research.
    Simulates realistic channel conditions, user traffic, and mobility patterns.
    """

    def __init__(self,
                 n_users: int = 20,
                 n_rbs: int = 50,  # Resource blocks
                 bandwidth_per_rb: float = 180e3,  # Hz
                 noise_power_dbm: float = -174,  # dBm/Hz
                 max_tx_power_dbm: float = 43,  # dBm
                 seed: int = 42):
        """
        Initialize the wireless data generator.

        Args:
            n_users: Number of users in the system
            n_rbs: Number of resource blocks
            bandwidth_per_rb: Bandwidth per resource block in Hz
            noise_power_dbm: Noise power spectral density in dBm/Hz
            max_tx_power_dbm: Maximum transmit power in dBm
            seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.bandwidth_per_rb = bandwidth_per_rb
        self.noise_power_dbm = noise_power_dbm
        self.max_tx_power_dbm = max_tx_power_dbm

        np.random.seed(seed)

        # Convert to linear scale
        self.noise_power_linear = self._dbm_to_linear(noise_power_dbm) * bandwidth_per_rb
        self.max_tx_power_linear = self._dbm_to_linear(max_tx_power_dbm)

    @staticmethod
    def _dbm_to_linear(dbm_value: float) -> float:
        """Convert dBm to linear scale (watts)"""
        return 10 ** ((dbm_value - 30) / 10)

    @staticmethod
    def _linear_to_db(linear_value: float) -> float:
        """Convert linear scale to dB"""
        return 10 * np.log10(np.maximum(linear_value, 1e-12))

    def generate_channel_gains(self,
                              n_time_slots: int,
                              scenario: str = 'urban_macro') -> np.ndarray:
        """
        Generate time-varying channel gains for all users and resource blocks.

        Args:
            n_time_slots: Number of time slots
            scenario: Channel scenario ('urban_macro', 'urban_micro', 'rural')

        Returns:
            Channel gains array of shape (n_time_slots, n_users, n_rbs)
        """

        # Scenario-specific parameters
        if scenario == 'urban_macro':
            path_loss_exp = 3.76
            shadow_std = 8.0  # dB
            k_factor_db = 3.0  # Rician K-factor
        elif scenario == 'urban_micro':
            path_loss_exp = 3.19
            shadow_std = 6.5
            k_factor_db = 9.0
        else:  # rural
            path_loss_exp = 2.7
            shadow_std = 4.0
            k_factor_db = 15.0

        # Generate user distances (100m to 1000m)
        distances = np.random.uniform(100, 1000, self.n_users)

        # Path loss calculation (simplified)
        path_loss_db = 32.4 + 20 * np.log10(2.0) + path_loss_exp * np.log10(distances / 1000)
        path_loss_linear = 10 ** (-path_loss_db / 10)

        # Generate correlated fading across time
        channel_gains = np.zeros((n_time_slots, self.n_users, self.n_rbs))

        for user in range(self.n_users):
            # Shadow fading (log-normal, slowly varying)
            shadow_fade_db = np.random.normal(0, shadow_std)
            shadow_fade_linear = 10 ** (shadow_fade_db / 10)

            for rb in range(self.n_rbs):
                # Fast fading (Rician for LOS, Rayleigh for NLOS)
                if np.random.random() < 0.3:  # 30% LOS probability
                    # Rician fading
                    k_linear = 10 ** (k_factor_db / 10)
                    fading = rice.rvs(np.sqrt(k_linear / (k_linear + 1)),
                                    scale=np.sqrt(1 / (2 * (k_linear + 1))),
                                    size=n_time_slots)
                    fading = fading ** 2  # Power
                else:
                    # Rayleigh fading
                    fading = rayleigh.rvs(scale=1/np.sqrt(2), size=n_time_slots)
                    fading = fading ** 2  # Power

                # Apply temporal correlation
                alpha = 0.9  # Correlation factor
                correlated_fading = np.zeros(n_time_slots)
                correlated_fading[0] = fading[0]
                for t in range(1, n_time_slots):
                    correlated_fading[t] = (alpha * correlated_fading[t-1] +
                                          np.sqrt(1 - alpha**2) * fading[t])

                # Combine path loss, shadow fading, and fast fading
                channel_gains[:, user, rb] = (path_loss_linear[user] *
                                            shadow_fade_linear *
                                            np.abs(correlated_fading))

        return channel_gains

    def generate_traffic_demands(self,
                                n_time_slots: int,
                                traffic_model: str = 'bursty') -> np.ndarray:
        """
        Generate time-varying traffic demands for all users.

        Args:
            n_time_slots: Number of time slots
            traffic_model: Traffic model ('constant', 'bursty', 'periodic')

        Returns:
            Traffic demands in Mbps, shape (n_time_slots, n_users)
        """

        if traffic_model == 'constant':
            # Constant traffic with small variations
            base_demands = np.random.uniform(1, 10, self.n_users)
            demands = np.tile(base_demands, (n_time_slots, 1))
            demands += np.random.normal(0, 0.1, (n_time_slots, self.n_users))

        elif traffic_model == 'bursty':
            # Bursty traffic using Poisson arrivals
            demands = np.zeros((n_time_slots, self.n_users))
            for user in range(self.n_users):
                # Base arrival rate
                lambda_base = np.random.uniform(0.1, 0.5)

                for t in range(n_time_slots):
                    # Time-varying arrival rate
                    lambda_t = lambda_base * (1 + 0.5 * np.sin(2 * np.pi * t / 100))

                    # Number of arrivals
                    arrivals = np.random.poisson(lambda_t)

                    # Data size per arrival (exponential distribution)
                    if arrivals > 0:
                        demands[t, user] = np.sum(np.random.exponential(2.0, arrivals))
                    else:
                        demands[t, user] = 0.1  # Minimum demand

        else:  # periodic
            # Periodic traffic patterns
            demands = np.zeros((n_time_slots, self.n_users))
            for user in range(self.n_users):
                # Random period and phase
                period = np.random.uniform(50, 200)
                phase = np.random.uniform(0, 2*np.pi)
                base_demand = np.random.uniform(2, 8)

                time_vector = np.arange(n_time_slots)
                demands[:, user] = (base_demand *
                                  (1 + 0.8 * np.sin(2*np.pi*time_vector/period + phase)) +
                                  np.random.normal(0, 0.2, n_time_slots))

        return np.maximum(demands, 0.1)  # Ensure minimum demand

    def generate_dataset(self,
                        n_time_slots: int = 1000,
                        scenario: str = 'urban_macro',
                        traffic_model: str = 'bursty') -> Dict[str, np.ndarray]:
        """
        Generate complete wireless dataset.

        Returns:
            Dictionary containing:
            - 'channel_gains': Channel gains (n_time_slots, n_users, n_rbs)
            - 'traffic_demands': Traffic demands in Mbps (n_time_slots, n_users)
            - 'sinr': Signal-to-interference-plus-noise ratio (n_time_slots, n_users, n_rbs)
            - 'timestamps': Time stamps
        """

        print(f"Generating wireless dataset...")
        print(f"Users: {self.n_users}, RBs: {self.n_rbs}, Time slots: {n_time_slots}")

        # Generate channel gains
        channel_gains = self.generate_channel_gains(n_time_slots, scenario)

        # Generate traffic demands
        traffic_demands = self.generate_traffic_demands(n_time_slots, traffic_model)

        # Calculate SINR (assuming equal power allocation initially)
        power_per_rb = self.max_tx_power_linear / self.n_rbs
        sinr = np.zeros_like(channel_gains)

        for t in range(n_time_slots):
            for user in range(self.n_users):
                for rb in range(self.n_rbs):
                    # Signal power
                    signal_power = power_per_rb * channel_gains[t, user, rb]

                    # Interference from other users on same RB (simplified)
                    interference = 0
                    for other_user in range(self.n_users):
                        if other_user != user:
                            interference += (power_per_rb *
                                           channel_gains[t, other_user, rb] * 0.1)  # 10% interference

                    # SINR calculation
                    sinr[t, user, rb] = signal_power / (interference + self.noise_power_linear)

        # Generate timestamps
        timestamps = np.arange(n_time_slots)

        dataset = {
            'channel_gains': channel_gains,
            'traffic_demands': traffic_demands,
            'sinr': sinr,
            'timestamps': timestamps,
            'metadata': {
                'n_users': self.n_users,
                'n_rbs': self.n_rbs,
                'scenario': scenario,
                'traffic_model': traffic_model,
                'bandwidth_per_rb': self.bandwidth_per_rb,
                'noise_power_dbm': self.noise_power_dbm,
                'max_tx_power_dbm': self.max_tx_power_dbm
            }
        }

        print("Dataset generation completed!")
        return dataset

    def save_dataset(self, dataset: Dict, filename: str):
        """Save dataset to file"""
        np.savez_compressed(filename, **dataset)
        print(f"Dataset saved to {filename}")

    def load_dataset(self, filename: str) -> Dict:
        """Load dataset from file"""
        data = np.load(filename, allow_pickle=True)
        dataset = {key: data[key] for key in data.keys()}
        print(f"Dataset loaded from {filename}")
        return dataset

    def plot_sample_data(self, dataset: Dict, save_path: Optional[str] = None):
        """Plot sample data for visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Channel gains over time for first user
        axes[0, 0].plot(dataset['timestamps'][:200],
                       self._linear_to_db(dataset['channel_gains'][:200, 0, 0]))
        axes[0, 0].set_title('Channel Gain (dB) - User 1, RB 1')
        axes[0, 0].set_xlabel('Time Slot')
        axes[0, 0].set_ylabel('Channel Gain (dB)')

        # Traffic demands over time
        axes[0, 1].plot(dataset['timestamps'][:200],
                       dataset['traffic_demands'][:200, :5])
        axes[0, 1].set_title('Traffic Demands - First 5 Users')
        axes[0, 1].set_xlabel('Time Slot')
        axes[0, 1].set_ylabel('Demand (Mbps)')
        axes[0, 1].legend([f'User {i+1}' for i in range(5)])

        # SINR distribution
        sinr_db = self._linear_to_db(dataset['sinr'].flatten())
        axes[1, 0].hist(sinr_db, bins=50, alpha=0.7)
        axes[1, 0].set_title('SINR Distribution')
        axes[1, 0].set_xlabel('SINR (dB)')
        axes[1, 0].set_ylabel('Frequency')

        # Average channel gain per user
        avg_gains = np.mean(dataset['channel_gains'], axis=(0, 2))
        axes[1, 1].bar(range(len(avg_gains)), self._linear_to_db(avg_gains))
        axes[1, 1].set_title('Average Channel Gain per User')
        axes[1, 1].set_xlabel('User ID')
        axes[1, 1].set_ylabel('Average Gain (dB)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Example usage
    generator = WirelessDataGenerator(n_users=10, n_rbs=25)

    # Generate dataset
    dataset = generator.generate_dataset(n_time_slots=500,
                                       scenario='urban_macro',
                                       traffic_model='bursty')

    # Save dataset
    generator.save_dataset(dataset, '../data/wireless_dataset.npz')

    # Plot sample data
    generator.plot_sample_data(dataset, '../figures/sample_data.png')