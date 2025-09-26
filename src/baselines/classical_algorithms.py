import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class ClassicalResourceAllocator:
    """
    Implements classical resource allocation algorithms for wireless networks.
    Includes Round Robin, Proportional Fair, Water-Filling, and Convex optimization.
    """

    def __init__(self,
                 n_users: int,
                 n_rbs: int,
                 max_power: float,
                 noise_power: float,
                 bandwidth_per_rb: float = 180e3):
        """
        Initialize the classical resource allocator.

        Args:
            n_users: Number of users
            n_rbs: Number of resource blocks
            max_power: Maximum transmit power (linear scale)
            noise_power: Noise power per RB (linear scale)
            bandwidth_per_rb: Bandwidth per resource block in Hz
        """
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.max_power = max_power
        self.noise_power = noise_power
        self.bandwidth_per_rb = bandwidth_per_rb

        # Initialize historical throughput for PF
        self.historical_throughput = np.ones(n_users)

    def round_robin(self,
                   channel_gains: np.ndarray,
                   time_slot: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Round Robin resource allocation.

        Args:
            channel_gains: Channel gains (n_users, n_rbs)
            time_slot: Current time slot

        Returns:
            Tuple of (resource_allocation, power_allocation)
        """
        # Initialize allocations
        rb_allocation = np.zeros((self.n_users, self.n_rbs), dtype=bool)
        power_allocation = np.zeros((self.n_users, self.n_rbs))

        # Simple Round Robin: assign RBs cyclically
        for rb in range(self.n_rbs):
            user = (time_slot + rb) % self.n_users
            rb_allocation[user, rb] = True
            power_allocation[user, rb] = self.max_power / self.n_rbs

        return rb_allocation, power_allocation

    def proportional_fair(self,
                         channel_gains: np.ndarray,
                         alpha: float = 1.0,
                         beta: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proportional Fair resource allocation.

        Args:
            channel_gains: Channel gains (n_users, n_rbs)
            alpha: PF fairness parameter
            beta: Throughput averaging factor

        Returns:
            Tuple of (resource_allocation, power_allocation)
        """
        # Initialize allocations
        rb_allocation = np.zeros((self.n_users, self.n_rbs), dtype=bool)
        power_allocation = np.zeros((self.n_users, self.n_rbs))

        # Calculate instantaneous rates
        power_per_rb = self.max_power / self.n_rbs
        sinr = channel_gains * power_per_rb / self.noise_power
        rates = self.bandwidth_per_rb * np.log2(1 + sinr) / 1e6  # Mbps

        # PF metric: rate / average_throughput^alpha
        pf_metric = rates / (self.historical_throughput.reshape(-1, 1) ** alpha)

        # Assign each RB to user with highest PF metric
        for rb in range(self.n_rbs):
            best_user = np.argmax(pf_metric[:, rb])
            rb_allocation[best_user, rb] = True
            power_allocation[best_user, rb] = power_per_rb

        # Update historical throughput
        current_throughput = np.sum(rates * rb_allocation, axis=1)
        self.historical_throughput = (beta * self.historical_throughput +
                                    (1 - beta) * current_throughput)

        return rb_allocation, power_allocation

    def water_filling(self,
                     channel_gains: np.ndarray,
                     user_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Water-filling power allocation with greedy RB assignment.

        Args:
            channel_gains: Channel gains (n_users, n_rbs)
            user_weights: Weights for each user

        Returns:
            Tuple of (resource_allocation, power_allocation)
        """
        if user_weights is None:
            user_weights = np.ones(self.n_users)

        # Initialize allocations
        rb_allocation = np.zeros((self.n_users, self.n_rbs), dtype=bool)
        power_allocation = np.zeros((self.n_users, self.n_rbs))

        # Greedy RB assignment based on channel quality
        for rb in range(self.n_rbs):
            # Find user with best weighted channel gain
            weighted_gains = channel_gains[:, rb] * user_weights
            best_user = np.argmax(weighted_gains)
            rb_allocation[best_user, rb] = True

        # Water-filling power allocation for each user
        for user in range(self.n_users):
            assigned_rbs = np.where(rb_allocation[user, :])[0]
            if len(assigned_rbs) == 0:
                continue

            # Water-filling across assigned RBs
            gains = channel_gains[user, assigned_rbs]
            n_assigned = len(assigned_rbs)

            # Water level calculation
            sorted_indices = np.argsort(gains)
            sorted_gains = gains[sorted_indices]

            total_power = self.max_power
            water_level = 0

            for k in range(n_assigned):
                # Try water level that fills k+1 RBs
                temp_level = (total_power + np.sum(self.noise_power / sorted_gains[:k+1])) / (k+1)

                if k == n_assigned - 1 or temp_level >= self.noise_power / sorted_gains[k+1]:
                    water_level = temp_level
                    n_filled = k + 1
                    break

            # Allocate power
            for i, rb_idx in enumerate(assigned_rbs):
                gain = channel_gains[user, rb_idx]
                if i < n_filled:
                    power_allocation[user, rb_idx] = max(0, water_level - self.noise_power / gain)

        return rb_allocation, power_allocation

    def convex_optimization(self,
                          channel_gains: np.ndarray,
                          traffic_demands: np.ndarray,
                          objective: str = 'sum_rate') -> Tuple[np.ndarray, np.ndarray]:
        """
        Convex optimization-based resource allocation.

        Args:
            channel_gains: Channel gains (n_users, n_rbs)
            traffic_demands: Traffic demands in Mbps (n_users,)
            objective: Optimization objective ('sum_rate', 'weighted_sum_rate', 'energy_efficiency')

        Returns:
            Tuple of (resource_allocation, power_allocation)
        """
        try:
            # Decision variables
            # Binary RB allocation
            x = cp.Variable((self.n_users, self.n_rbs), boolean=True)
            # Power allocation
            p = cp.Variable((self.n_users, self.n_rbs), nonneg=True)

            # Constraints
            constraints = []

            # Each RB assigned to at most one user
            for rb in range(self.n_rbs):
                constraints.append(cp.sum(x[:, rb]) <= 1)

            # Power constraint
            constraints.append(cp.sum(p) <= self.max_power)

            # Power only allocated to assigned RBs
            for user in range(self.n_users):
                for rb in range(self.n_rbs):
                    constraints.append(p[user, rb] <= self.max_power * x[user, rb])

            # Calculate rates (approximation for convex formulation)
            rates = cp.Variable((self.n_users, self.n_rbs), nonneg=True)
            for user in range(self.n_users):
                for rb in range(self.n_rbs):
                    # Linear approximation of log rate
                    sinr_linear = channel_gains[user, rb] * p[user, rb] / self.noise_power
                    # Use first-order approximation: log(1+x) ≈ x for small x
                    rate_approx = self.bandwidth_per_rb * sinr_linear / (1e6 * np.log(2))
                    constraints.append(rates[user, rb] <= rate_approx * x[user, rb])

            # Objective function
            if objective == 'sum_rate':
                objective_fn = cp.Maximize(cp.sum(rates))
            elif objective == 'weighted_sum_rate':
                weights = traffic_demands / np.sum(traffic_demands)
                objective_fn = cp.Maximize(cp.sum(cp.multiply(weights.reshape(-1, 1), rates)))
            elif objective == 'energy_efficiency':
                total_rate = cp.sum(rates)
                total_power = cp.sum(p)
                objective_fn = cp.Maximize(total_rate - 0.1 * total_power)  # EE approximation

            # Solve
            problem = cp.Problem(objective_fn, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status not in ["infeasible", "unbounded"]:
                rb_allocation = x.value > 0.5
                power_allocation = p.value
                # Ensure non-negative values
                power_allocation = np.maximum(power_allocation, 0)
            else:
                # Fallback to proportional fair
                print(f"Convex optimization failed: {problem.status}. Using Proportional Fair.")
                return self.proportional_fair(channel_gains)

        except Exception as e:
            print(f"Convex optimization error: {e}. Using Proportional Fair.")
            return self.proportional_fair(channel_gains)

        return rb_allocation.astype(bool), power_allocation

    def calculate_metrics(self,
                         rb_allocation: np.ndarray,
                         power_allocation: np.ndarray,
                         channel_gains: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics for given allocation.

        Args:
            rb_allocation: RB allocation matrix (n_users, n_rbs)
            power_allocation: Power allocation matrix (n_users, n_rbs)
            channel_gains: Channel gains (n_users, n_rbs)

        Returns:
            Dictionary of performance metrics
        """
        # Calculate SINR and rates
        sinr = np.zeros_like(channel_gains)
        rates = np.zeros_like(channel_gains)

        for user in range(self.n_users):
            for rb in range(self.n_rbs):
                if rb_allocation[user, rb]:
                    # Signal power
                    signal_power = power_allocation[user, rb] * channel_gains[user, rb]

                    # Interference (simplified model)
                    interference = 0
                    for other_user in range(self.n_users):
                        if other_user != user and rb_allocation[other_user, rb]:
                            interference += (power_allocation[other_user, rb] *
                                           channel_gains[other_user, rb] * 0.1)

                    # SINR
                    sinr[user, rb] = signal_power / (interference + self.noise_power)

                    # Rate (Shannon capacity)
                    rates[user, rb] = (self.bandwidth_per_rb *
                                     np.log2(1 + sinr[user, rb]) / 1e6)  # Mbps

        # Aggregate metrics
        user_rates = np.sum(rates, axis=1)
        total_rate = np.sum(user_rates)
        total_power = np.sum(power_allocation)

        # Fairness index (Jain's fairness)
        fairness = (np.sum(user_rates) ** 2) / (self.n_users * np.sum(user_rates ** 2))
        if np.isnan(fairness):
            fairness = 0

        # Energy efficiency (bits per joule)
        energy_efficiency = total_rate / max(total_power, 1e-10)

        # Resource utilization
        resource_utilization = np.sum(rb_allocation) / self.n_rbs

        metrics = {
            'total_throughput_mbps': total_rate,
            'avg_user_throughput_mbps': np.mean(user_rates),
            'min_user_throughput_mbps': np.min(user_rates),
            'max_user_throughput_mbps': np.max(user_rates),
            'fairness_index': fairness,
            'total_power_watts': total_power,
            'energy_efficiency_mbps_per_watt': energy_efficiency,
            'resource_utilization': resource_utilization,
            'avg_sinr_db': 10 * np.log10(np.mean(sinr[sinr > 0])) if np.any(sinr > 0) else -np.inf
        }

        return metrics

    def benchmark_algorithms(self,
                            channel_gains: np.ndarray,
                            traffic_demands: np.ndarray,
                            algorithms: Optional[list] = None) -> Dict[str, Dict]:
        """
        Benchmark multiple algorithms on the same scenario.

        Args:
            channel_gains: Channel gains (n_users, n_rbs)
            traffic_demands: Traffic demands (n_users,)
            algorithms: List of algorithms to benchmark

        Returns:
            Dictionary of results for each algorithm
        """
        if algorithms is None:
            algorithms = ['round_robin', 'proportional_fair', 'water_filling', 'convex_optimization']

        results = {}

        for alg_name in algorithms:
            print(f"Running {alg_name}...")
            start_time = time.time()

            try:
                if alg_name == 'round_robin':
                    rb_alloc, power_alloc = self.round_robin(channel_gains, 0)
                elif alg_name == 'proportional_fair':
                    rb_alloc, power_alloc = self.proportional_fair(channel_gains)
                elif alg_name == 'water_filling':
                    rb_alloc, power_alloc = self.water_filling(channel_gains)
                elif alg_name == 'convex_optimization':
                    rb_alloc, power_alloc = self.convex_optimization(channel_gains, traffic_demands)

                execution_time = time.time() - start_time
                metrics = self.calculate_metrics(rb_alloc, power_alloc, channel_gains)
                metrics['execution_time_ms'] = execution_time * 1000

                results[alg_name] = {
                    'metrics': metrics,
                    'rb_allocation': rb_alloc,
                    'power_allocation': power_alloc
                }

            except Exception as e:
                print(f"Error in {alg_name}: {e}")
                results[alg_name] = {'error': str(e)}

        return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # System parameters
    n_users = 10
    n_rbs = 20
    max_power = 1.0  # 1 Watt
    noise_power = 1e-10  # 0.1 nW

    # Generate sample data
    channel_gains = np.random.exponential(1e-8, (n_users, n_rbs))  # Exponential fading
    traffic_demands = np.random.uniform(1, 5, n_users)  # 1-5 Mbps demands

    # Initialize allocator
    allocator = ClassicalResourceAllocator(n_users, n_rbs, max_power, noise_power)

    # Benchmark algorithms
    results = allocator.benchmark_algorithms(channel_gains, traffic_demands)

    # Print results
    print("\nBenchmark Results:")
    print("-" * 50)
    for alg_name, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"\n{alg_name.upper()}:")
            print(f"  Total Throughput: {metrics['total_throughput_mbps']:.2f} Mbps")
            print(f"  Fairness Index: {metrics['fairness_index']:.3f}")
            print(f"  Energy Efficiency: {metrics['energy_efficiency_mbps_per_watt']:.2f} Mbps/W")
            print(f"  Execution Time: {metrics['execution_time_ms']:.2f} ms")