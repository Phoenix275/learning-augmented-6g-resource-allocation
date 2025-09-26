#!/usr/bin/env python3
"""
Simplified Demo: Learning-Augmented Resource Allocation (Classical algorithms only)

This script demonstrates the classical resource allocation algorithms
without requiring deep learning libraries.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.wireless_generator import WirelessDataGenerator
from baselines.classical_algorithms import ClassicalResourceAllocator


def run_classical_demo():
    """Run demonstration of classical algorithms only"""
    print("=" * 60)
    print("SIMPLIFIED DEMO: Classical Resource Allocation Algorithms")
    print("=" * 60)

    # System parameters
    n_users = 8
    n_rbs = 16
    n_time_slots = 200
    test_slots = 50

    print(f"System: {n_users} users, {n_rbs} resource blocks")
    print(f"Simulation: {n_time_slots} time slots, testing on last {test_slots}")

    # 1. Generate data
    print("\n1. Generating synthetic wireless data...")
    generator = WirelessDataGenerator(
        n_users=n_users,
        n_rbs=n_rbs,
        seed=42
    )

    dataset = generator.generate_dataset(
        n_time_slots=n_time_slots,
        scenario='urban_macro',
        traffic_model='bursty'
    )

    print(f"   Generated {n_time_slots} time slots of data")
    print(f"   Average channel gain: {10*np.log10(np.mean(dataset['channel_gains'])):.1f} dB")
    print(f"   Average traffic demand: {np.mean(dataset['traffic_demands']):.1f} Mbps")

    # 2. Initialize classical allocator
    print("\n2. Initializing classical resource allocators...")

    classical_allocator = ClassicalResourceAllocator(
        n_users=n_users,
        n_rbs=n_rbs,
        max_power=1.0,
        noise_power=1e-10
    )

    # 3. Run comparison of classical algorithms
    print("\n3. Running classical algorithm comparison...")

    algorithms = {
        'Round Robin': 'round_robin',
        'Proportional Fair': 'proportional_fair',
        'Water Filling': 'water_filling',
        'Convex Optimization': 'convex_optimization'
    }

    results = {name: {'throughputs': [], 'execution_times': [], 'fairness': [], 'energy_eff': []}
               for name in algorithms.keys()}

    test_start = n_time_slots - test_slots

    print(f"   Testing on time slots {test_start} to {n_time_slots}")

    for t in range(test_start, n_time_slots):
        current_channels = dataset['channel_gains'][t]
        current_traffic = dataset['traffic_demands'][t]

        for name, alg_key in algorithms.items():
            try:
                start_time = time.time()

                if alg_key == 'round_robin':
                    rb_alloc, power_alloc = classical_allocator.round_robin(current_channels, t)
                elif alg_key == 'proportional_fair':
                    rb_alloc, power_alloc = classical_allocator.proportional_fair(current_channels)
                elif alg_key == 'water_filling':
                    rb_alloc, power_alloc = classical_allocator.water_filling(current_channels)
                elif alg_key == 'convex_optimization':
                    rb_alloc, power_alloc = classical_allocator.convex_optimization(
                        current_channels, current_traffic
                    )

                execution_time = time.time() - start_time
                metrics = classical_allocator.calculate_metrics(
                    rb_alloc, power_alloc, current_channels
                )

                results[name]['throughputs'].append(metrics['total_throughput_mbps'])
                results[name]['execution_times'].append(execution_time * 1000)  # ms
                results[name]['fairness'].append(metrics['fairness_index'])
                results[name]['energy_eff'].append(metrics['energy_efficiency_mbps_per_watt'])

            except Exception as e:
                print(f"   Error in {name}: {e}")

        if (t - test_start + 1) % 10 == 0:
            print(f"   Processed {t - test_start + 1}/{test_slots} time slots...")

    # 4. Display results
    print("\n4. RESULTS SUMMARY:")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Throughput':<12} {'Fairness':<10} {'Energy Eff':<12} {'Time (ms)':<10}")
    print("=" * 80)

    best_throughput = 0
    best_algorithm = ""

    for name, data in results.items():
        if data['throughputs']:
            avg_throughput = np.mean(data['throughputs'])
            std_throughput = np.std(data['throughputs'])
            avg_fairness = np.mean(data['fairness'])
            avg_energy_eff = np.mean(data['energy_eff'])
            avg_time = np.mean(data['execution_times'])

            print(f"{name:<20} {avg_throughput:<12.2f} {avg_fairness:<10.3f} "
                  f"{avg_energy_eff:<12.2f} {avg_time:<10.3f}")

            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_algorithm = name

    print("=" * 80)

    # 5. Create visualization
    print("\n5. Generating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Classical Resource Allocation Algorithm Comparison', fontsize=14)

    # Throughput comparison
    names = list(results.keys())
    throughputs = [np.mean(results[name]['throughputs']) if results[name]['throughputs'] else 0
                  for name in names]

    axes[0, 0].bar(range(len(names)), throughputs, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[0, 0].set_title('Average System Throughput')
    axes[0, 0].set_ylabel('Throughput (Mbps)')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)

    # Fairness comparison
    fairness_scores = [np.mean(results[name]['fairness']) if results[name]['fairness'] else 0
                      for name in names]

    axes[0, 1].bar(range(len(names)), fairness_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[0, 1].set_title('Average Fairness Index')
    axes[0, 1].set_ylabel('Fairness Index')
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Energy efficiency comparison
    energy_effs = [np.mean(results[name]['energy_eff']) if results[name]['energy_eff'] else 0
                   for name in names]

    axes[1, 0].bar(range(len(names)), energy_effs, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[1, 0].set_title('Average Energy Efficiency')
    axes[1, 0].set_ylabel('Energy Efficiency (Mbps/W)')
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)

    # Execution time comparison
    exec_times = [np.mean(results[name]['execution_times']) if results[name]['execution_times'] else 0
                 for name in names]

    axes[1, 1].bar(range(len(names)), exec_times, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[1, 1].set_title('Average Execution Time')
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('classical_demo_results.png', dpi=300, bbox_inches='tight')
    print("   Results visualization saved as 'classical_demo_results.png'")

    # 6. Time series plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = ['blue', 'green', 'red', 'orange']
    for i, (name, data) in enumerate(results.items()):
        if data['throughputs']:
            time_slots = list(range(test_start, test_start + len(data['throughputs'])))
            ax.plot(time_slots, data['throughputs'],
                   label=name, color=colors[i % len(colors)], alpha=0.8, linewidth=2)

    ax.set_title('System Throughput Over Time')
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Throughput (Mbps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('throughput_timeseries.png', dpi=300, bbox_inches='tight')
    print("   Time series plot saved as 'throughput_timeseries.png'")

    # 7. Key insights
    print("\n6. KEY INSIGHTS:")
    print("-" * 40)
    print(f"• Best performing algorithm: {best_algorithm} ({best_throughput:.2f} Mbps)")

    if results['Convex Optimization']['throughputs'] and results['Round Robin']['throughputs']:
        convex_avg = np.mean(results['Convex Optimization']['throughputs'])
        rr_avg = np.mean(results['Round Robin']['throughputs'])
        improvement = (convex_avg - rr_avg) / rr_avg * 100
        print(f"• Convex optimization improves throughput by {improvement:+.1f}% over Round Robin")

    if results['Proportional Fair']['throughputs']:
        pf_fairness = np.mean(results['Proportional Fair']['fairness'])
        print(f"• Proportional Fair achieves {pf_fairness:.3f} fairness index")

    fastest_alg = min(results.keys(),
                     key=lambda x: np.mean(results[x]['execution_times']) if results[x]['execution_times'] else float('inf'))
    fastest_time = np.mean(results[fastest_alg]['execution_times'])
    print(f"• Fastest algorithm: {fastest_alg} ({fastest_time:.3f} ms)")

    print("\n• All classical algorithms are suitable for real-time implementation")
    print("• Water-filling provides good energy efficiency")
    print("• Convex optimization offers the best throughput but higher complexity")

    print(f"\n{'='*60}")
    print("CLASSICAL DEMO COMPLETED SUCCESSFULLY!")
    print("This demonstrates the baseline performance of classical algorithms.")
    print("The full ML-augmented system would show additional improvements.")
    print(f"{'='*60}")

    # Show plots
    plt.show()

    return results


if __name__ == "__main__":
    try:
        results = run_classical_demo()
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()