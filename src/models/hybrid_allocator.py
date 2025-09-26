import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional, Union
import time
import warnings
from src.baselines.classical_algorithms import ClassicalResourceAllocator
from src.models.prediction_models import WirelessPredictor

warnings.filterwarnings('ignore')


class HybridMLAllocator:
    """
    Hybrid ML-Augmented Resource Allocator that combines machine learning predictions
    with classical optimization algorithms for enhanced performance and efficiency.
    """

    def __init__(self,
                 n_users: int,
                 n_rbs: int,
                 max_power: float,
                 noise_power: float,
                 bandwidth_per_rb: float = 180e3,
                 predictor: Optional[WirelessPredictor] = None,
                 classical_fallback: str = 'proportional_fair'):
        """
        Initialize the hybrid ML-augmented allocator.

        Args:
            n_users: Number of users
            n_rbs: Number of resource blocks
            max_power: Maximum transmit power (linear scale)
            noise_power: Noise power per RB (linear scale)
            bandwidth_per_rb: Bandwidth per resource block in Hz
            predictor: Trained ML predictor (optional)
            classical_fallback: Fallback algorithm when ML fails
        """
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.max_power = max_power
        self.noise_power = noise_power
        self.bandwidth_per_rb = bandwidth_per_rb
        self.predictor = predictor
        self.classical_fallback = classical_fallback

        # Initialize classical allocator for fallback and comparison
        self.classical_allocator = ClassicalResourceAllocator(
            n_users, n_rbs, max_power, noise_power, bandwidth_per_rb
        )

        # Prediction confidence tracking
        self.prediction_history = []
        self.confidence_threshold = 0.7

    def set_predictor(self, predictor: WirelessPredictor):
        """Set or update the ML predictor"""
        self.predictor = predictor

    def predict_warm_start(self,
                          channel_history: np.ndarray,
                          traffic_history: np.ndarray,
                          current_channels: np.ndarray,
                          current_traffic: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions and warm start values for optimization.

        Args:
            channel_history: Historical channel gains
            traffic_history: Historical traffic demands
            current_channels: Current channel gains
            current_traffic: Current traffic demands

        Returns:
            Dictionary containing predictions and warm start values
        """
        results = {
            'predicted_channels': current_channels,
            'predicted_traffic': current_traffic,
            'confidence_score': 0.0,
            'warm_start_allocation': None,
            'warm_start_power': None,
            'use_prediction': False
        }

        if self.predictor is None or not self.predictor.is_trained:
            return results

        try:
            # Make predictions
            pred_channels, pred_traffic = self.predictor.predict(
                channel_history, traffic_history
            )

            # Use first prediction step
            if len(pred_channels.shape) > 2:
                pred_channels_next = pred_channels[0]
                pred_traffic_next = pred_traffic[0]
            else:
                pred_channels_next = pred_channels
                pred_traffic_next = pred_traffic

            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(
                current_channels, current_traffic,
                pred_channels_next, pred_traffic_next
            )

            # Generate warm start allocation based on predictions
            if confidence > self.confidence_threshold:
                warm_rb, warm_power = self._generate_warm_start_allocation(
                    pred_channels_next, pred_traffic_next
                )

                results.update({
                    'predicted_channels': pred_channels_next,
                    'predicted_traffic': pred_traffic_next,
                    'confidence_score': confidence,
                    'warm_start_allocation': warm_rb,
                    'warm_start_power': warm_power,
                    'use_prediction': True
                })

        except Exception as e:
            print(f"Prediction error: {e}")

        return results

    def _calculate_prediction_confidence(self,
                                       current_channels: np.ndarray,
                                       current_traffic: np.ndarray,
                                       pred_channels: np.ndarray,
                                       pred_traffic: np.ndarray) -> float:
        """
        Calculate confidence score for predictions based on historical accuracy
        and prediction stability.
        """
        try:
            # Channel prediction error (normalized RMSE)
            channel_error = np.sqrt(np.mean((current_channels - pred_channels) ** 2))
            channel_range = np.max(current_channels) - np.min(current_channels)
            channel_nrmse = channel_error / (channel_range + 1e-10)

            # Traffic prediction error (normalized RMSE)
            traffic_error = np.sqrt(np.mean((current_traffic - pred_traffic) ** 2))
            traffic_range = np.max(current_traffic) - np.min(current_traffic)
            traffic_nrmse = traffic_error / (traffic_range + 1e-10)

            # Combined confidence (inverse of normalized error)
            confidence = 1.0 / (1.0 + 0.5 * (channel_nrmse + traffic_nrmse))

            # Update prediction history for adaptive confidence
            self.prediction_history.append({
                'channel_nrmse': channel_nrmse,
                'traffic_nrmse': traffic_nrmse,
                'confidence': confidence
            })

            # Keep only recent history
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]

            # Adaptive confidence based on recent performance
            if len(self.prediction_history) > 10:
                recent_confidences = [h['confidence'] for h in self.prediction_history[-10:]]
                adaptive_factor = np.mean(recent_confidences)
                confidence = confidence * adaptive_factor

            return np.clip(confidence, 0.0, 1.0)

        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return 0.0

    def _generate_warm_start_allocation(self,
                                      pred_channels: np.ndarray,
                                      pred_traffic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate warm start allocation based on predictions using fast heuristic.
        """
        try:
            # Use predicted values with proportional fair as warm start
            rb_allocation, power_allocation = self.classical_allocator.proportional_fair(
                pred_channels, alpha=0.8  # Slightly less aggressive fairness
            )
            return rb_allocation, power_allocation

        except Exception as e:
            print(f"Warm start generation error: {e}")
            # Fallback to simple allocation
            rb_allocation = np.zeros((self.n_users, self.n_rbs), dtype=bool)
            power_allocation = np.zeros((self.n_users, self.n_rbs))

            # Round-robin allocation as fallback
            for rb in range(self.n_rbs):
                user = rb % self.n_users
                rb_allocation[user, rb] = True
                power_allocation[user, rb] = self.max_power / self.n_rbs

            return rb_allocation, power_allocation

    def ml_augmented_convex_optimization(self,
                                       current_channels: np.ndarray,
                                       current_traffic: np.ndarray,
                                       prediction_results: Dict[str, np.ndarray],
                                       objective: str = 'weighted_sum_rate') -> Tuple[np.ndarray, np.ndarray]:
        """
        ML-augmented convex optimization with warm start and prediction-informed constraints.
        """
        try:
            # Extract prediction information
            pred_channels = prediction_results.get('predicted_channels', current_channels)
            pred_traffic = prediction_results.get('predicted_traffic', current_traffic)
            warm_start_rb = prediction_results.get('warm_start_allocation')
            warm_start_power = prediction_results.get('warm_start_power')
            use_prediction = prediction_results.get('use_prediction', False)

            # Decision variables
            x = cp.Variable((self.n_users, self.n_rbs), boolean=True)  # RB allocation
            p = cp.Variable((self.n_users, self.n_rbs), nonneg=True)   # Power allocation

            # Constraints
            constraints = []

            # Each RB assigned to at most one user
            for rb in range(self.n_rbs):
                constraints.append(cp.sum(x[:, rb]) <= 1)

            # Total power constraint
            constraints.append(cp.sum(p) <= self.max_power)

            # Power only allocated to assigned RBs
            for user in range(self.n_users):
                for rb in range(self.n_rbs):
                    constraints.append(p[user, rb] <= self.max_power * x[user, rb])

            # If using predictions, add prediction-informed constraints
            if use_prediction:
                confidence = prediction_results.get('confidence_score', 0.0)

                # Encourage allocation to users with high predicted traffic
                high_traffic_users = pred_traffic > np.mean(pred_traffic)
                traffic_weights = pred_traffic / np.sum(pred_traffic)

                # Add soft constraints for high-traffic users
                for user in range(self.n_users):
                    if high_traffic_users[user]:
                        min_rbs = max(1, int(traffic_weights[user] * self.n_rbs * 0.5))
                        constraints.append(cp.sum(x[user, :]) >= min_rbs * confidence)

            # Calculate rates with prediction-aware channel estimates
            if use_prediction:
                # Blend current and predicted channels based on confidence
                conf = prediction_results.get('confidence_score', 0.0)
                effective_channels = (conf * pred_channels +
                                    (1 - conf) * current_channels)
            else:
                effective_channels = current_channels

            # Approximate rates (linear approximation for convexity)
            rates = cp.Variable((self.n_users, self.n_rbs), nonneg=True)
            for user in range(self.n_users):
                for rb in range(self.n_rbs):
                    # Linear approximation of Shannon capacity
                    # log(1 + SNR) ≈ SNR for low SNR, bounded for high SNR
                    snr_coeff = effective_channels[user, rb] / self.noise_power
                    rate_upper_bound = self.bandwidth_per_rb * snr_coeff * p[user, rb] / (1e6 * np.log(2))
                    constraints.append(rates[user, rb] <= rate_upper_bound * x[user, rb])

            # Objective function
            if objective == 'sum_rate':
                objective_fn = cp.Maximize(cp.sum(rates))
            elif objective == 'weighted_sum_rate':
                # Weight by traffic demands
                if use_prediction:
                    weights = pred_traffic / np.sum(pred_traffic)
                else:
                    weights = current_traffic / np.sum(current_traffic)
                objective_fn = cp.Maximize(cp.sum(cp.multiply(weights.reshape(-1, 1), rates)))
            elif objective == 'energy_efficiency':
                total_rate = cp.sum(rates)
                total_power = cp.sum(p)
                objective_fn = cp.Maximize(total_rate - 0.01 * total_power)

            # Warm start if available
            if warm_start_rb is not None and warm_start_power is not None:
                x.value = warm_start_rb.astype(float)
                p.value = warm_start_power

            # Solve
            problem = cp.Problem(objective_fn, constraints)
            problem.solve(solver=cp.ECOS, verbose=False, warm_start=True)

            if problem.status not in ["infeasible", "unbounded"]:
                rb_allocation = x.value > 0.5
                power_allocation = np.maximum(p.value, 0)
            else:
                print(f"Optimization failed: {problem.status}. Using classical fallback.")
                return self._classical_fallback(current_channels, current_traffic)

        except Exception as e:
            print(f"ML-augmented optimization error: {e}. Using classical fallback.")
            return self._classical_fallback(current_channels, current_traffic)

        return rb_allocation.astype(bool), power_allocation

    def ml_guided_proportional_fair(self,
                                  current_channels: np.ndarray,
                                  current_traffic: np.ndarray,
                                  prediction_results: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        ML-guided proportional fair allocation that uses predictions to adjust fairness parameters.
        """
        try:
            use_prediction = prediction_results.get('use_prediction', False)
            confidence = prediction_results.get('confidence_score', 0.0)

            if use_prediction and confidence > self.confidence_threshold:
                pred_traffic = prediction_results['predicted_traffic']
                pred_channels = prediction_results['predicted_channels']

                # Adaptive fairness parameter based on predicted traffic variation
                traffic_cv = np.std(pred_traffic) / np.mean(pred_traffic)
                alpha = 1.0 + 0.5 * traffic_cv * confidence  # More fairness for high traffic variation

                # Blend current and predicted channels
                effective_channels = (confidence * pred_channels +
                                    (1 - confidence) * current_channels)

                # Use guided proportional fair
                rb_allocation, power_allocation = self.classical_allocator.proportional_fair(
                    effective_channels, alpha=alpha
                )
            else:
                # Standard proportional fair
                rb_allocation, power_allocation = self.classical_allocator.proportional_fair(
                    current_channels
                )

        except Exception as e:
            print(f"ML-guided PF error: {e}. Using classical fallback.")
            return self._classical_fallback(current_channels, current_traffic)

        return rb_allocation, power_allocation

    def _classical_fallback(self,
                           current_channels: np.ndarray,
                           current_traffic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback to classical algorithm when ML fails"""
        if self.classical_fallback == 'proportional_fair':
            return self.classical_allocator.proportional_fair(current_channels)
        elif self.classical_fallback == 'water_filling':
            return self.classical_allocator.water_filling(current_channels)
        else:  # round_robin
            return self.classical_allocator.round_robin(current_channels, 0)

    def allocate_resources(self,
                          channel_history: np.ndarray,
                          traffic_history: np.ndarray,
                          current_channels: np.ndarray,
                          current_traffic: np.ndarray,
                          method: str = 'ml_augmented_convex',
                          **kwargs) -> Dict[str, np.ndarray]:
        """
        Main resource allocation method that combines ML predictions with optimization.

        Args:
            channel_history: Historical channel gains
            traffic_history: Historical traffic demands
            current_channels: Current channel gains
            current_traffic: Current traffic demands
            method: Allocation method ('ml_augmented_convex', 'ml_guided_pf', 'classical')

        Returns:
            Dictionary containing allocation results and metadata
        """
        start_time = time.time()

        # Generate predictions and warm start
        prediction_results = self.predict_warm_start(
            channel_history, traffic_history, current_channels, current_traffic
        )

        # Perform resource allocation
        if method == 'ml_augmented_convex':
            rb_allocation, power_allocation = self.ml_augmented_convex_optimization(
                current_channels, current_traffic, prediction_results, **kwargs
            )
        elif method == 'ml_guided_pf':
            rb_allocation, power_allocation = self.ml_guided_proportional_fair(
                current_channels, current_traffic, prediction_results
            )
        else:  # classical
            rb_allocation, power_allocation = self._classical_fallback(
                current_channels, current_traffic
            )

        execution_time = time.time() - start_time

        # Calculate metrics
        metrics = self.classical_allocator.calculate_metrics(
            rb_allocation, power_allocation, current_channels
        )
        metrics['execution_time_ms'] = execution_time * 1000

        return {
            'rb_allocation': rb_allocation,
            'power_allocation': power_allocation,
            'metrics': metrics,
            'prediction_results': prediction_results,
            'method_used': method
        }

    def benchmark_methods(self,
                         channel_history: np.ndarray,
                         traffic_history: np.ndarray,
                         current_channels: np.ndarray,
                         current_traffic: np.ndarray,
                         methods: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Benchmark different allocation methods"""

        if methods is None:
            methods = ['classical', 'ml_guided_pf', 'ml_augmented_convex']

        results = {}
        for method in methods:
            print(f"Benchmarking {method}...")
            try:
                result = self.allocate_resources(
                    channel_history, traffic_history,
                    current_channels, current_traffic,
                    method=method
                )
                results[method] = result
            except Exception as e:
                print(f"Error in {method}: {e}")
                results[method] = {'error': str(e)}

        return results


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.data.wireless_generator import WirelessDataGenerator
    from src.models.prediction_models import WirelessPredictor

    # Generate sample data
    generator = WirelessDataGenerator(n_users=8, n_rbs=16)
    dataset = generator.generate_dataset(n_time_slots=500)

    # Train a predictor
    predictor = WirelessPredictor(
        n_users=8, n_rbs=16, model_type='lstm',
        sequence_length=10, prediction_horizon=1
    )

    # Prepare and train
    train_loader, test_loader = predictor.prepare_data(
        dataset['channel_gains'], dataset['traffic_demands']
    )
    predictor.train(train_loader, test_loader, epochs=20)

    # Initialize hybrid allocator
    hybrid_allocator = HybridMLAllocator(
        n_users=8, n_rbs=16,
        max_power=1.0, noise_power=1e-10,
        predictor=predictor
    )

    # Test allocation
    history_len = 10
    channels_hist = dataset['channel_gains'][-history_len-1:-1]
    traffic_hist = dataset['traffic_demands'][-history_len-1:-1]
    current_channels = dataset['channel_gains'][-1]
    current_traffic = dataset['traffic_demands'][-1]

    # Benchmark methods
    results = hybrid_allocator.benchmark_methods(
        channels_hist, traffic_hist, current_channels, current_traffic
    )

    # Print results
    print("\nHybrid Allocator Benchmark Results:")
    print("-" * 60)
    for method, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            pred_results = result.get('prediction_results', {})
            print(f"\n{method.upper()}:")
            print(f"  Total Throughput: {metrics['total_throughput_mbps']:.2f} Mbps")
            print(f"  Energy Efficiency: {metrics['energy_efficiency_mbps_per_watt']:.2f} Mbps/W")
            print(f"  Execution Time: {metrics['execution_time_ms']:.2f} ms")
            if 'confidence_score' in pred_results:
                print(f"  Prediction Confidence: {pred_results['confidence_score']:.3f}")
                print(f"  Used Prediction: {pred_results['use_prediction']}")
        else:
            print(f"{method}: ERROR - {result['error']}")