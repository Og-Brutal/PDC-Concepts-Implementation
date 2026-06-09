"""
Section B - Question 3: All-to-All Personalized Communication on a d-Dimensional Hypercube
Implements dimension-ordered routing with double buffering and barrier synchronization.
"""

import time
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict 

# ============================================================
# Hypercube Node
# ============================================================

class HypercubeNode:
    """Represents a node in the hypercube topology."""
    
    def __init__(self, node_id, d):
        self.id = node_id
        self.d = d
        self.p = 2 ** d
        # Each node has personalized messages for every other node
        # msg[j] = (source, destination) tuple
        self.messages = {}
        for j in range(self.p):
            if j != node_id:
                self.messages[j] = (node_id, j)
        self.lock = threading.Lock()
    
    def get_neighbors(self):
        """Return neighbors (differ by exactly 1 bit)."""
        return [self.id ^ (1 << bit) for bit in range(self.d)]
    
    def count_delivered(self):
        """Count messages that have reached their final destination (this node)."""
        return sum(1 for dest, (src, dst) in self.messages.items() if dst == self.id)


# ============================================================
# Hypercube All-to-All Algorithm
# ============================================================

def hypercube_alltoall_simulation(d):
    """
    Simulate all-to-all personalized communication on a d-dimensional hypercube.
    
    Algorithm: d-step dimension-ordered exchange
    - At step k (0-indexed), node i exchanges messages with neighbor whose ID 
      differs in bit k
    - After exchange, each node keeps messages whose destination matches its 
      own ID in the lowest k+1 bits
    - Uses double buffering: send buffer and receive buffer
    
    Returns: nodes, step_data (messages in transit per step), total_time
    """
    p = 2 ** d
    
    # Initialize nodes
    # node_msgs[i] = list of (src, dest) tuples for all messages node i holds
    node_msgs = {}
    for i in range(p):
        node_msgs[i] = [(i, j) for j in range(p) if j != i]
    
    step_data = []  # Messages not at final destination per step
    step_times = []
    
    print(f"\nHypercube All-to-All: d={d}, p={p}")
    print(f"{'='*50}")
    
    # Count initial undelivered
    initial_undelivered = sum(len(node_msgs[i]) for i in range(p))
    print(f"Initial messages in transit: {initial_undelivered}")
    
    total_start = time.time()
    
    for step in range(d):
        step_start = time.time()
        
        # For each pair of nodes differing in bit 'step', exchange messages
        new_node_msgs = {i: [] for i in range(p)}
        
        for i in range(p):
            partner = i ^ (1 << step)
            
            # Determine which messages to keep and which to send to partner
            for (src, dest) in node_msgs[i]:
                # If dest's bit at position 'step' matches partner's bit, send to partner
                if (dest >> step) & 1 == (partner >> step) & 1:
                    new_node_msgs[partner].append((src, dest))
                else:
                    # Keep this message (dest's bit matches our bit at this position)
                    new_node_msgs[i].append((src, dest))
        
        node_msgs = new_node_msgs
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Count messages not at final destination
        undelivered = 0
        for i in range(p):
            for (src, dest) in node_msgs[i]:
                if dest != i:
                    undelivered += 1
        
        step_data.append(undelivered)
        print(f"Step {step} (dim {step}): partner bit flip at position {step}")
        print(f"  Messages still in transit: {undelivered}")
    
    total_time = time.time() - total_start
    
    return node_msgs, step_data, total_time, step_times


def verify_hypercube_result(node_msgs, d):
    """Verify that each node has received exactly the messages destined for it."""
    p = 2 ** d
    correct = True
    
    for i in range(p):
        # Node i should have messages (j, i) for all j != i
        expected_sources = set(range(p)) - {i}
        received_sources = set()
        
        for (src, dst) in node_msgs[i]:
            if dst == i:
                received_sources.add(src)
            else:
                print(f"ERROR: Node {i} holds message ({src},{dst}) which isn't for it!")
                correct = False
        
        if received_sources != expected_sources:
            missing = expected_sources - received_sources
            if missing:
                print(f"ERROR: Node {i} missing messages from: {missing}")
                correct = False
    
    return correct


# ============================================================
# Threaded Implementation
# ============================================================

def hypercube_alltoall_threaded(d):
    """
    Multi-threaded implementation with barriers and mutexes.
    """
    p = 2 ** d
    
    # Shared data structures
    node_msgs = {}
    for i in range(p):
        node_msgs[i] = [(i, j) for j in range(p) if j != i]
    
    node_msgs_next = {i: [] for i in range(p)}
    barrier = threading.Barrier(p)
    global_lock = threading.Lock()
    
    step_data = [0] * d
    
    def worker(node_id):
        nonlocal node_msgs, node_msgs_next
        
        for step in range(d):
            partner = node_id ^ (1 << step)
            
            # Prepare send and keep buffers
            to_send = []
            to_keep = []
            
            for (src, dest) in node_msgs[node_id]:
                if (dest >> step) & 1 == (partner >> step) & 1:
                    to_send.append((src, dest))
                else:
                    to_keep.append((src, dest))
            
            barrier.wait()  # Ensure all nodes have prepared their buffers
            
            # Write to buffers (protected by lock)
            with global_lock:
                node_msgs_next[node_id].extend(to_keep)
                node_msgs_next[partner].extend(to_send)
            
            barrier.wait()  # Ensure all exchanges complete
            
            # Swap buffers (only one thread does this)
            if node_id == 0:
                for i in range(p):
                    node_msgs[i] = node_msgs_next[i]
                    node_msgs_next[i] = []
                
                # Count undelivered
                undelivered = sum(1 for ni in range(p) for (s, d_id) in node_msgs[ni] if d_id != ni)
                step_data[step] = undelivered
            
            barrier.wait()  # Ensure swap is complete
    
    threads = []
    for i in range(p):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
    
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start_time
    
    return node_msgs, step_data, elapsed


# ============================================================
# Performance Analysis & Plotting
# ============================================================

def run_performance_analysis():
    """Run for multiple dimensions and analyze."""
    
    dimensions = [3, 4, 5, 6]
    results = {}
    
    print("\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)
    
    print(f"\n{'Dimension d':<12} {'Nodes p':<10} {'Steps':<8} {'Critical Path':<15} {'Time (ms)'}")
    print("-" * 60)
    
    for d in dimensions:
        p = 2 ** d
        node_msgs, step_data, total_time, step_times = hypercube_alltoall_simulation(d)
        
        correct = verify_hypercube_result(node_msgs, d)
        print(f"\nHypercube all-to-all: {'CORRECT' if correct else 'INCORRECT'}")
        
        results[d] = {
            'p': p,
            'steps': d,
            'step_data': step_data,
            'total_time': total_time,
            'step_times': step_times,
        }
        
        print(f"{d:<12} {p:<10} {d:<8} {d:<15} {total_time*1000:.2f}")
    
    # Message complexity analysis
    print(f"\n{'='*50}")
    print("Message Complexity Analysis")
    print(f"{'='*50}")
    for d in dimensions:
        p = 2 ** d
        msgs_per_node = p - 1  # Each node sends p-1 messages total
        global_msgs = p * (p - 1)  # Total messages globally
        print(f"d={d}, p={p}: Messages per node={msgs_per_node}, Global={global_msgs}")
    
    return results


def plot_results(results):
    """Generate plots for performance analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Messages in transit per step (for d=4)
    ax = axes[0]
    for d in sorted(results.keys()):
        step_data = results[d]['step_data']
        ax.plot(range(1, len(step_data) + 1), step_data, 'o-', 
                label=f'd={d}, p={results[d]["p"]}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Messages Not at Final Destination')
    ax.set_title('Messages in Transit per Step')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Execution time vs dimension/nodes
    ax = axes[1]
    dims = sorted(results.keys())
    p_values = [results[d]['p'] for d in dims]
    times = [results[d]['total_time'] * 1000 for d in dims]
    
    ax.plot(p_values, times, 's-', color='#6366f1', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes (p = 2^d)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Total Execution Time vs Number of Nodes')
    ax.grid(alpha=0.3)
    
    # Add log scale for x-axis
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('hypercube_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: hypercube_performance.png")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("All-to-All Personalized Communication on d-Dimensional Hypercube")
    print("=" * 60)
    
    # Example: d=3 (8 nodes)
    d = 3
    print(f"\n--- Running for d={d}, p={2**d} ---")
    node_msgs, step_data, total_time, step_times = hypercube_alltoall_simulation(d)
    
    # Verify correctness
    correct = verify_hypercube_result(node_msgs, d)
    print(f"\nHypercube all-to-all: {'CORRECT' if correct else 'INCORRECT'}")
    
    # Show final messages per node
    p = 2 ** d
    print(f"\nFinal message distribution:")
    for i in range(p):
        delivered = [(src, dst) for (src, dst) in node_msgs[i] if dst == i]
        print(f"  Node {i}: received {len(delivered)} messages from nodes "
              f"{sorted([s for s,_ in delivered])}")
    
    # Run full performance analysis
    results = run_performance_analysis()
    
    # Plot
    plot_results(results)
    
    # Written analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("""
The number of in-flight messages decreases each step because at step k, 
every node routes messages based on bit k of the destination address. After 
step k, all messages are in nodes whose lower k+1 bits match their 
destination's lower k+1 bits. This progressive "bit-matching" means each 
step resolves one dimension of routing, halving the potential wrong-node 
placements.

The hypercube structure leads to logarithmic critical-path length (d = log2(p)
steps) because each dimension step independently resolves one bit of 
addressing. Unlike a ring (p-1 steps) or mesh (2*sqrt(p) steps), the hypercube's 
exponential connectivity allows each step to exchange with a partner that 
differs in exactly one bit, effectively doubling the "known" address bits 
per step. This gives excellent scalability: doubling the nodes only adds 
one more step. The trade-off is that message sizes grow (each node exchanges
p/2 messages per step), increasing link bandwidth requirements. However, 
the O(log p) critical path makes hypercube ideal for latency-sensitive 
collective operations.
""")
