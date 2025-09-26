import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.wireless_generator import WirelessDataGenerator
from src.baselines.classical_algorithms import ClassicalResourceAllocator
from src.models.prediction_models import WirelessPredictor
from src.models.hybrid_allocator import HybridMLAllocator


class WirelessSimulator:
    """
    Comprehensive simulation environment for evaluating resource allocation algorithms
    in wireless networks. Supports multiple scenarios, metrics, and comparison studies.
    """

    def __init__(self,
                 n_users: int = 20,
                 n_rbs: int = 50,
                 max_power: float = 1.0,
                 noise_power_dbm: float = -174,
                 bandwidth_per_rb: float = 180e3,
                 scenario: str = 'urban_macro',
                 traffic_model: str = 'bursty',
                 seed: int = 42):
        """
        Initialize the wireless simulator.

        Args:
            n_users: Number of users in the system
            n_rbs: Number of resource blocks
            max_power: Maximum transmit power in watts
            noise_power_dbm: Noise power spectral density in dBm/Hz
            bandwidth_per_rb: Bandwidth per resource block in Hz
            scenario: Channel scenario ('urban_macro', 'urban_micro', 'rural')
            traffic_model: Traffic model ('constant', 'bursty', 'periodic')
            seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.max_power = max_power
        self.noise_power_dbm = noise_power_dbm
        self.bandwidth_per_rb = bandwidth_per_rb
        self.scenario = scenario
        self.traffic_model = traffic_model

        # Convert noise power to linear scale
        self.noise_power = self._dbm_to_linear(noise_power_dbm) * bandwidth_per_rb

        # Initialize components
        self.data_generator = WirelessDataGenerator(
            n_users=n_users,
            n_rbs=n_rbs,
            bandwidth_per_rb=bandwidth_per_rb,
            noise_power_dbm=noise_power_dbm,
            max_tx_power_dbm=self._linear_to_dbm(max_power),
            seed=seed
        )

        self.classical_allocator = ClassicalResourceAllocator(
            n_users=n_users,
            n_rbs=n_rbs,
            max_power=max_power,
            noise_power=self.noise_power,
            bandwidth_per_rb=bandwidth_per_rb
        )

        # Results storage
        self.simulation_results = {}
        self.performance_metrics = {}

    @staticmethod
    def _dbm_to_linear(dbm_value: float) -> float:
        """Convert dBm to linear scale (watts)"""
        return 10 ** ((dbm_value - 30) / 10)

    @staticmethod
    def _linear_to_dbm(linear_value: float) -> float:
        """Convert linear scale to dBm"""
        return 30 + 10 * np.log10(linear_value)

    def generate_scenario(self,
                         n_time_slots: int = 1000,
                         save_data: bool = True) -> Dict[str, np.ndarray]:
        """Generate a complete simulation scenario"""
        print(f"Generating scenario: {self.scenario} with {self.traffic_model} traffic")
        print(f"Parameters: {self.n_users} users, {self.n_rbs} RBs, {n_time_slots} time slots")

        dataset = self.data_generator.generate_dataset(
            n_time_slots=n_time_slots,
            scenario=self.scenario,
            traffic_model=self.traffic_model
        )

        if save_data:
            filename = f"scenario_{self.scenario}_{self.traffic_model}_{n_time_slots}.npz"
            filepath = os.path.join('../../results', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.data_generator.save_dataset(dataset, filepath)

        return dataset

    def train_ml_predictor(self,
                          dataset: Dict[str, np.ndarray],
                          model_type: str = 'lstm',
                          sequence_length: int = 10,
                          prediction_horizon: int = 1,
                          epochs: int = 50,
                          train_ratio: float = 0.8) -> WirelessPredictor:
        """Train ML predictor on the dataset"""
        print(f"Training {model_type} predictor...")

        predictor = WirelessPredictor(
            n_users=self.n_users,
            n_rbs=self.n_rbs,
            model_type=model_type,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )

        # Prepare training data
        train_loader, test_loader = predictor.prepare_data(
            dataset['channel_gains'],
            dataset['traffic_demands'],
            test_size=1-train_ratio
        )

        # Train the model
        training_history = predictor.train(
            train_loader,
            test_loader,
            epochs=epochs,
            patience=15
        )

        # Save model
        model_filename = f"predictor_{model_type}_{self.scenario}_{self.traffic_model}.pth"
        model_path = os.path.join('../../results', model_filename)
        predictor.save_model(model_path)

        print(f"Predictor training completed. Final test loss: {training_history['test_losses'][-1]:.6f}")
        return predictor

    def run_algorithm_comparison(self,
                                dataset: Dict[str, np.ndarray],
                                predictor: Optional[WirelessPredictor] = None,
                                algorithms: Optional[List[str]] = None,
                                n_test_slots: int = 200,
                                history_length: int = 10) -> Dict[str, List[Dict]]:
        """
        Run comprehensive comparison of different resource allocation algorithms.

        Args:
            dataset: Generated wireless dataset
            predictor: Trained ML predictor (optional)
            algorithms: List of algorithms to compare
            n_test_slots: Number of time slots for testing
            history_length: Length of historical data for ML methods

        Returns:
            Dictionary of results for each algorithm
        """
        if algorithms is None:
            algorithms = ['round_robin', 'proportional_fair', 'water_filling', 'convex_optimization']
            if predictor is not None:
                algorithms.extend(['ml_guided_pf', 'ml_augmented_convex'])

        print(f"Running algorithm comparison for {len(algorithms)} algorithms over {n_test_slots} time slots...")

        # Initialize hybrid allocator if predictor is available
        hybrid_allocator = None
        if predictor is not None:
            hybrid_allocator = HybridMLAllocator(
                n_users=self.n_users,
                n_rbs=self.n_rbs,
                max_power=self.max_power,
                noise_power=self.noise_power,
                predictor=predictor
            )

        # Extract test data
        total_slots = dataset['channel_gains'].shape[0]
        test_start = max(history_length, total_slots - n_test_slots)

        results = {alg: [] for alg in algorithms}
        progress_bar = tqdm(range(test_start, total_slots), desc="Simulating")

        for t in progress_bar:
            current_channels = dataset['channel_gains'][t]
            current_traffic = dataset['traffic_demands'][t]

            # Historical data for ML methods
            if t >= history_length:
                channel_history = dataset['channel_gains'][t-history_length:t]
                traffic_history = dataset['traffic_demands'][t-history_length:t]
            else:
                channel_history = None
                traffic_history = None

            for alg in algorithms:
                try:
                    start_time = time.time()

                    if alg in ['round_robin', 'proportional_fair', 'water_filling', 'convex_optimization']:
                        # Classical algorithms
                        if alg == 'round_robin':
                            rb_alloc, power_alloc = self.classical_allocator.round_robin(
                                current_channels, t
                            )
                        elif alg == 'proportional_fair':
                            rb_alloc, power_alloc = self.classical_allocator.proportional_fair(
                                current_channels
                            )
                        elif alg == 'water_filling':
                            rb_alloc, power_alloc = self.classical_allocator.water_filling(
                                current_channels
                            )
                        elif alg == 'convex_optimization':
                            rb_alloc, power_alloc = self.classical_allocator.convex_optimization(
                                current_channels, current_traffic
                            )

                        metrics = self.classical_allocator.calculate_metrics(
                            rb_alloc, power_alloc, current_channels
                        )

                    elif alg in ['ml_guided_pf', 'ml_augmented_convex'] and hybrid_allocator is not None:
                        # ML-augmented algorithms
                        if channel_history is not None and traffic_history is not None:
                            result = hybrid_allocator.allocate_resources(
                                channel_history, traffic_history,
                                current_channels, current_traffic,
                                method=alg
                            )
                            metrics = result['metrics']
                        else:
                            # Fallback to classical if no history
                            rb_alloc, power_alloc = self.classical_allocator.proportional_fair(
                                current_channels
                            )
                            metrics = self.classical_allocator.calculate_metrics(
                                rb_alloc, power_alloc, current_channels
                            )

                    execution_time = time.time() - start_time
                    metrics['execution_time_ms'] = execution_time * 1000

                    # Store results
                    results[alg].append({
                        'time_slot': t,
                        'metrics': metrics.copy()
                    })

                except Exception as e:
                    print(f"Error in {alg} at time {t}: {e}")
                    # Store error result
                    results[alg].append({
                        'time_slot': t,
                        'error': str(e)
                    })

            # Update progress bar with current performance
            if len(results[algorithms[0]]) > 0 and 'metrics' in results[algorithms[0]][-1]:
                current_throughput = results[algorithms[0]][-1]['metrics']['total_throughput_mbps']
                progress_bar.set_postfix({'Throughput': f'{current_throughput:.1f} Mbps'})

        print("Algorithm comparison completed!")
        return results

    def calculate_aggregate_metrics(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Calculate aggregate performance metrics from simulation results"""
        print("Calculating aggregate metrics...")

        metrics_data = []

        for alg_name, time_series in results.items():
            # Filter out error results
            valid_results = [r for r in time_series if 'metrics' in r]
            if not valid_results:
                continue

            # Extract metrics
            throughputs = [r['metrics']['total_throughput_mbps'] for r in valid_results]
            fairness_indices = [r['metrics']['fairness_index'] for r in valid_results]
            energy_efficiencies = [r['metrics']['energy_efficiency_mbps_per_watt'] for r in valid_results]
            execution_times = [r['metrics']['execution_time_ms'] for r in valid_results]
            avg_sinrs = [r['metrics']['avg_sinr_db'] for r in valid_results if r['metrics']['avg_sinr_db'] != -np.inf]

            # Calculate statistics
            metrics_data.append({
                'Algorithm': alg_name,
                'Avg_Throughput_Mbps': np.mean(throughputs),
                'Std_Throughput_Mbps': np.std(throughputs),
                'Min_Throughput_Mbps': np.min(throughputs),
                'Max_Throughput_Mbps': np.max(throughputs),
                'Median_Throughput_Mbps': np.median(throughputs),
                'Avg_Fairness_Index': np.mean(fairness_indices),
                'Std_Fairness_Index': np.std(fairness_indices),
                'Avg_Energy_Efficiency': np.mean(energy_efficiencies),
                'Std_Energy_Efficiency': np.std(energy_efficiencies),
                'Avg_Execution_Time_ms': np.mean(execution_times),
                'Std_Execution_Time_ms': np.std(execution_times),
                'Avg_SINR_dB': np.mean(avg_sinrs) if avg_sinrs else 0,
                'Success_Rate': len(valid_results) / len(time_series),
                'Total_Samples': len(time_series)
            })

        df = pd.DataFrame(metrics_data)
        return df

    def generate_comparison_plots(self,
                                 results: Dict[str, List[Dict]],
                                 aggregate_metrics: pd.DataFrame,
                                 save_path: Optional[str] = None):
        """Generate comprehensive comparison plots"""
        print("Generating comparison plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Resource Allocation Algorithm Comparison\n'
                    f'Scenario: {self.scenario}, Traffic: {self.traffic_model}',
                    fontsize=16)

        # 1. Average Throughput Comparison
        ax = axes[0, 0]
        sns.barplot(data=aggregate_metrics, x='Algorithm', y='Avg_Throughput_Mbps', ax=ax)
        ax.set_title('Average System Throughput')
        ax.set_ylabel('Throughput (Mbps)')
        ax.tick_params(axis='x', rotation=45)

        # 2. Fairness Index Comparison
        ax = axes[0, 1]
        sns.barplot(data=aggregate_metrics, x='Algorithm', y='Avg_Fairness_Index', ax=ax)
        ax.set_title('Average Fairness Index')
        ax.set_ylabel('Fairness Index')
        ax.tick_params(axis='x', rotation=45)

        # 3. Energy Efficiency Comparison
        ax = axes[0, 2]
        sns.barplot(data=aggregate_metrics, x='Algorithm', y='Avg_Energy_Efficiency', ax=ax)
        ax.set_title('Average Energy Efficiency')
        ax.set_ylabel('Energy Efficiency (Mbps/W)')
        ax.tick_params(axis='x', rotation=45)

        # 4. Execution Time Comparison
        ax = axes[1, 0]
        sns.barplot(data=aggregate_metrics, x='Algorithm', y='Avg_Execution_Time_ms', ax=ax)
        ax.set_title('Average Execution Time')
        ax.set_ylabel('Execution Time (ms)')
        ax.tick_params(axis='x', rotation=45)

        # 5. Throughput Distribution (Box Plot)
        ax = axes[1, 1]
        throughput_data = []
        for alg_name, time_series in results.items():
            valid_results = [r for r in time_series if 'metrics' in r]
            throughputs = [r['metrics']['total_throughput_mbps'] for r in valid_results]
            throughput_data.extend([(alg_name, t) for t in throughputs])

        if throughput_data:
            throughput_df = pd.DataFrame(throughput_data, columns=['Algorithm', 'Throughput'])
            sns.boxplot(data=throughput_df, x='Algorithm', y='Throughput', ax=ax)
            ax.set_title('Throughput Distribution')
            ax.set_ylabel('Throughput (Mbps)')
            ax.tick_params(axis='x', rotation=45)

        # 6. Time Series Plot (Sample algorithms)
        ax = axes[1, 2]
        sample_algorithms = list(results.keys())[:3]  # Show first 3 algorithms
        for alg_name in sample_algorithms:
            valid_results = [r for r in results[alg_name] if 'metrics' in r]
            if valid_results:
                time_slots = [r['time_slot'] for r in valid_results[:100]]  # First 100 samples
                throughputs = [r['metrics']['total_throughput_mbps'] for r in valid_results[:100]]
                ax.plot(time_slots, throughputs, label=alg_name, alpha=0.7)

        ax.set_title('Throughput Over Time (Sample)')
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Throughput (Mbps)')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")

        plt.show()

    def run_complete_evaluation(self,
                               n_time_slots: int = 1000,
                               n_test_slots: int = 200,
                               model_types: List[str] = ['lstm'],
                               classical_algorithms: Optional[List[str]] = None,
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline including data generation,
        model training, and algorithm comparison.
        """
        if classical_algorithms is None:
            classical_algorithms = ['round_robin', 'proportional_fair', 'water_filling']

        print("=" * 60)
        print("Starting Complete Wireless Resource Allocation Evaluation")
        print("=" * 60)

        evaluation_results = {
            'scenario': self.scenario,
            'traffic_model': self.traffic_model,
            'system_parameters': {
                'n_users': self.n_users,
                'n_rbs': self.n_rbs,
                'max_power': self.max_power,
                'noise_power_dbm': self.noise_power_dbm
            },
            'simulation_results': {},
            'aggregate_metrics': {},
            'summary': {}
        }

        # Step 1: Generate scenario data
        dataset = self.generate_scenario(n_time_slots)

        # Step 2: Train ML predictors
        predictors = {}
        for model_type in model_types:
            print(f"\nTraining {model_type} predictor...")
            predictor = self.train_ml_predictor(
                dataset,
                model_type=model_type,
                epochs=30
            )
            predictors[model_type] = predictor

        # Step 3: Run algorithm comparisons
        all_algorithms = classical_algorithms.copy()

        for model_type, predictor in predictors.items():
            print(f"\nRunning comparison with {model_type} predictor...")

            # Add ML-augmented algorithms
            ml_algorithms = [f'ml_guided_pf_{model_type}', f'ml_augmented_convex_{model_type}']
            current_algorithms = classical_algorithms + ml_algorithms

            results = self.run_algorithm_comparison(
                dataset, predictor, current_algorithms, n_test_slots
            )

            # Calculate aggregate metrics
            aggregate_metrics = self.calculate_aggregate_metrics(results)

            # Store results
            evaluation_results['simulation_results'][model_type] = results
            evaluation_results['aggregate_metrics'][model_type] = aggregate_metrics

            # Generate plots
            plot_filename = f'comparison_{model_type}_{self.scenario}_{self.traffic_model}.png'
            plot_path = os.path.join('../../figures', plot_filename)
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

            self.generate_comparison_plots(results, aggregate_metrics, plot_path)

        # Step 4: Generate summary
        self._generate_evaluation_summary(evaluation_results)

        # Step 5: Save results
        if save_results:
            results_filename = f'evaluation_{self.scenario}_{self.traffic_model}.json'
            results_path = os.path.join('../../results', results_filename)

            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(evaluation_results)

            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"\nComplete results saved to {results_path}")

        return evaluation_results

    def _generate_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Generate evaluation summary"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        summary = {}

        for model_type, metrics_df in evaluation_results['aggregate_metrics'].items():
            print(f"\n{model_type.upper()} Model Results:")
            print("-" * 40)

            # Find best performing algorithm for each metric
            best_throughput = metrics_df.loc[metrics_df['Avg_Throughput_Mbps'].idxmax()]
            best_fairness = metrics_df.loc[metrics_df['Avg_Fairness_Index'].idxmax()]
            best_efficiency = metrics_df.loc[metrics_df['Avg_Energy_Efficiency'].idxmax()]
            fastest = metrics_df.loc[metrics_df['Avg_Execution_Time_ms'].idxmin()]

            print(f"Best Throughput: {best_throughput['Algorithm']} ({best_throughput['Avg_Throughput_Mbps']:.2f} Mbps)")
            print(f"Best Fairness: {best_fairness['Algorithm']} ({best_fairness['Avg_Fairness_Index']:.3f})")
            print(f"Best Energy Efficiency: {best_efficiency['Algorithm']} ({best_efficiency['Avg_Energy_Efficiency']:.2f} Mbps/W)")
            print(f"Fastest: {fastest['Algorithm']} ({fastest['Avg_Execution_Time_ms']:.2f} ms)")

            summary[model_type] = {
                'best_throughput': {'algorithm': best_throughput['Algorithm'],
                                   'value': float(best_throughput['Avg_Throughput_Mbps'])},
                'best_fairness': {'algorithm': best_fairness['Algorithm'],
                                 'value': float(best_fairness['Avg_Fairness_Index'])},
                'best_efficiency': {'algorithm': best_efficiency['Algorithm'],
                                   'value': float(best_efficiency['Avg_Energy_Efficiency'])},
                'fastest': {'algorithm': fastest['Algorithm'],
                           'value': float(fastest['Avg_Execution_Time_ms'])}
            }

        evaluation_results['summary'] = summary

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


if __name__ == "__main__":
    # Example usage
    print("Initializing Wireless Simulator...")

    # Create simulator instance
    simulator = WirelessSimulator(
        n_users=10,
        n_rbs=25,
        scenario='urban_macro',
        traffic_model='bursty'
    )

    # Run complete evaluation
    results = simulator.run_complete_evaluation(
        n_time_slots=500,
        n_test_slots=100,
        model_types=['lstm'],
        save_results=True
    )

    print("\nSimulation completed successfully!")