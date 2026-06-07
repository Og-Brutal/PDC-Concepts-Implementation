"""
Section B - Question 1: Topology-Aware All-to-All Personalized Communication
Implements and visualizes all-to-all personalized communication on Ring, 2D Mesh, and Hypercube.
"""

import time
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ============================================================
# Part A: Topology Construction & Visualization
# ============================================================

class Node:
    """Represents a processing node in any topology."""
    def __init__(self, node_id, total_nodes):
        self.id = node_id
        self.p = total_nodes
        # Initial messages: msg[j] = (source=id, dest=j) for all j != id
        self.messages = {j: (node_id, j) for j in range(total_nodes) if j != node_id}
        self.received = {}  # Messages that have reached their destination
        self.send_buffer = {}
        self.recv_buffer = {}
        self.lock = threading.Lock()

    def count_undelivered(self):
        """Count messages not yet at their final destination."""
        count = 0
        for dest, msg in self.messages.items():
            if dest != self.id:
                count += 1
        return count


def build_ring(p):
    """Build a ring topology: node i connected to (i+1) mod p and (i-1) mod p."""
    neighbors = {}
    for i in range(p):
        neighbors[i] = [(i + 1) % p, (i - 1 + p) % p]
    return neighbors


def build_mesh(p):
    """Build a 2D mesh topology: sqrt(p) x sqrt(p) grid."""
    sqrtP = int(np.sqrt(p))
    assert sqrtP * sqrtP == p, "p must be a perfect square for mesh"
    neighbors = {}
    for i in range(p):
        r, c = divmod(i, sqrtP)
        nbrs = []
        if c + 1 < sqrtP: nbrs.append(i + 1)      # right
        if c - 1 >= 0:    nbrs.append(i - 1)        # left
        if r + 1 < sqrtP: nbrs.append(i + sqrtP)    # down
        if r - 1 >= 0:    nbrs.append(i - sqrtP)    # up
        neighbors[i] = nbrs
    return neighbors


def build_hypercube(d):
    """Build a d-dimensional hypercube: nodes connected if IDs differ in exactly 1 bit."""
    p = 2 ** d
    neighbors = {}
    for i in range(p):
        nbrs = []
        for bit in range(d):
            nbrs.append(i ^ (1 << bit))
        neighbors[i] = nbrs
    return neighbors


def visualize_topology(neighbors, p, topology_name, pos=None):
    """Visualize a topology using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title(f"{topology_name} Topology (p={p})", fontsize=14, fontweight='bold')
    
    if pos is None:
        if topology_name == "Ring":
            angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
            pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
        elif topology_name == "2D Mesh":
            sqrtP = int(np.sqrt(p))
            pos = {i: (i % sqrtP, sqrtP - 1 - i // sqrtP) for i in range(p)}
        elif topology_name == "Hypercube":
            d = int(np.log2(p))
            if d <= 3:
                angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
                pos = {i: (np.cos(angles[i]) + 0.3 * (i >> (d-1)), 
                           np.sin(angles[i]) + 0.3 * (i >> (d-1))) for i in range(p)}
            else:
                angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
                pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(p)}
    
    # Draw edges
    drawn = set()
    for node, nbrs in neighbors.items():
        for nbr in nbrs:
            edge = tuple(sorted([node, nbr]))
            if edge not in drawn:
                drawn.add(edge)
                x = [pos[node][0], pos[nbr][0]]
                y = [pos[node][1], pos[nbr][1]]
                ax.plot(x, y, 'b-', alpha=0.3, linewidth=1)
    
    # Draw nodes
    for node in range(p):
        circle = plt.Circle(pos[node], 0.12, color='#6366f1', ec='white', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(pos[node][0], pos[node][1], str(node), ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=6)
    
    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.axis('off')
    return fig, pos


# ============================================================
# Part B: Algorithmic Implementations
# ============================================================

def ring_alltoall(p, m=1024):
    """
    Ring all-to-all personalized communication using p-1 step shift algorithm.
    In step k, node i sends the message intended for node (i+k) mod p to its neighbor.
    Returns: steps, messages_per_step
    """
    # Initialize: node i has messages for all other nodes
    node_data = {}
    for i in range(p):
        node_data[i] = {j: f"msg({i},{j})" for j in range(p) if j != i}
    
    steps = 0
    messages_in_transit = []
    
    for k in range(1, p):
        # In step k, each node sends to right neighbor (i+1) mod p
        new_data = {i: dict(node_data[i]) for i in range(p)}
        
        for i in range(p):
            # Node i sends message destined for (i+k) mod p to right neighbor
            dest = (i + k) % p
            right_nbr = (i + 1) % p
            
            if dest in node_data[i]:
                msg = node_data[i][dest]
                new_data[right_nbr][dest] = msg
                if dest != right_nbr:
                    pass  # Message still in transit
        
        node_data = new_data
        steps += 1
        
        # Count undelivered
        undelivered = 0
        for i in range(p):
            for dest in node_data[i]:
                if dest != i:
                    undelivered += 1
        messages_in_transit.append(undelivered)
    
    return steps, messages_in_transit


def mesh_alltoall(p, m=1024):
    """
    2D Mesh all-to-all personalized communication using two-phase routing.
    Phase 1: Row-wise all-to-all
    Phase 2: Column-wise all-to-all
    """
    sqrtP = int(np.sqrt(p))
    
    # Initialize: node i has messages for all other nodes
    node_data = {}
    for i in range(p):
        node_data[i] = {j: f"msg({i},{j})" for j in range(p) if j != i}
    
    steps = 0
    messages_in_transit = []
    
    # Phase 1: Row-wise communication (sqrt(p)-1 steps per row)
    for k in range(1, sqrtP):
        new_data = {i: dict(node_data[i]) for i in range(p)}
        
        for r in range(sqrtP):
            for c in range(sqrtP):
                i = r * sqrtP + c
                # Send to right neighbor in same row
                right_c = (c + 1) % sqrtP
                right_nbr = r * sqrtP + right_c
                
                # Send messages that need to go to right_nbr's column
                for dest, msg in list(node_data[i].items()):
                    dest_col = dest % sqrtP
                    if dest_col == right_c:
                        new_data[right_nbr][dest] = msg
        
        node_data = new_data
        steps += 1
        
        undelivered = sum(1 for i in range(p) for d in node_data[i] if d != i)
        messages_in_transit.append(undelivered)
    
    # Phase 2: Column-wise communication (sqrt(p)-1 steps per column)
    for k in range(1, sqrtP):
        new_data = {i: dict(node_data[i]) for i in range(p)}
        
        for r in range(sqrtP):
            for c in range(sqrtP):
                i = r * sqrtP + c
                # Send to down neighbor in same column
                down_r = (r + 1) % sqrtP
                down_nbr = down_r * sqrtP + c
                
                for dest, msg in list(node_data[i].items()):
                    if dest == down_nbr:
                        new_data[down_nbr][dest] = msg
        
        node_data = new_data
        steps += 1
        
        undelivered = sum(1 for i in range(p) for d in node_data[i] if d != i)
        messages_in_transit.append(undelivered)
    
    return steps, messages_in_transit


def hypercube_alltoall(d, m=1024):
    """
    Hypercube all-to-all personalized communication using dimension-ordered routing.
    d steps, each step exchanges across one dimension.
    """
    p = 2 ** d
    
    # Initialize: node i has messages for all other nodes
    node_data = {}
    for i in range(p):
        node_data[i] = {j: f"msg({i},{j})" for j in range(p) if j != i}
    
    steps = 0
    messages_in_transit = []
    
    for dim in range(d):
        new_data = {i: dict(node_data[i]) for i in range(p)}
        
        for i in range(p):
            partner = i ^ (1 << dim)
            # Send messages whose destination matches partner in bit 'dim'
            to_send = {}
            for dest, msg in list(node_data[i].items()):
                if (dest >> dim) & 1 != (i >> dim) & 1:
                    to_send[dest] = msg
            
            for dest, msg in to_send.items():
                new_data[partner][dest] = msg
        
        node_data = new_data
        steps += 1
        
        undelivered = sum(1 for i in range(p) for d_id in node_data[i] if d_id != i)
        messages_in_transit.append(undelivered)
    
    return steps, messages_in_transit


# ============================================================
# Part C: Performance & Comparative Analysis
# ============================================================

def run_comparison(p_values=[8, 16, 32]):
    """Run all three topologies and compare performance."""
    results = {}
    
    for p in p_values:
        d = int(np.log2(p))
        sqrtP = int(np.sqrt(p))
        
        print(f"\n{'='*50}")
        print(f"Running for p = {p} nodes")
        print(f"{'='*50}")
        
        # Ring
        t_start = time.time()
        ring_steps, ring_transit = ring_alltoall(p)
        ring_time = time.time() - t_start
        
        # Mesh (only if p is perfect square)
        if sqrtP * sqrtP == p:
            t_start = time.time()
            mesh_steps, mesh_transit = mesh_alltoall(p)
            mesh_time = time.time() - t_start
        else:
            mesh_steps, mesh_transit, mesh_time = None, None, None
        
        # Hypercube
        t_start = time.time()
        hyper_steps, hyper_transit = hypercube_alltoall(d)
        hyper_time = time.time() - t_start
        
        results[p] = {
            'ring': (ring_steps, ring_time, ring_transit),
            'mesh': (mesh_steps, mesh_time, mesh_transit) if mesh_steps else None,
            'hypercube': (hyper_steps, hyper_time, hyper_transit)
        }
        
        # Print summary table
        print(f"\n{'Topology':<12} {'Steps':<8} {'Complexity':<15} {'Time (ms)':<12}")
        print(f"{'-'*47}")
        print(f"{'Ring':<12} {ring_steps:<8} {'O(p-1)='+str(p-1):<15} {ring_time*1000:.2f}")
        if mesh_steps:
            print(f"{'Mesh':<12} {mesh_steps:<8} {'O(2(sqrtP-1))='+str(2*(sqrtP-1)):<15} {mesh_time*1000:.2f}")
        print(f"{'Hypercube':<12} {hyper_steps:<8} {'O(log2(p))='+str(d):<15} {hyper_time*1000:.2f}")
    
    return results


def plot_comparison(results):
    """Generate comparison bar chart and step analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart: Steps comparison
    ax = axes[0]
    p_values = sorted(results.keys())
    x = np.arange(len(p_values))
    width = 0.25
    
    ring_steps = [results[p]['ring'][0] for p in p_values]
    mesh_steps = [results[p]['mesh'][0] if results[p]['mesh'] else 0 for p in p_values]
    hyper_steps = [results[p]['hypercube'][0] for p in p_values]
    
    ax.bar(x - width, ring_steps, width, label='Ring', color='#f59e0b', alpha=0.85)
    ax.bar(x, mesh_steps, width, label='Mesh', color='#10b981', alpha=0.85)
    ax.bar(x + width, hyper_steps, width, label='Hypercube', color='#6366f1', alpha=0.85)
    
    ax.set_xlabel('Number of Nodes (p)')
    ax.set_ylabel('Communication Steps')
    ax.set_title('Steps Comparison: Ring vs Mesh vs Hypercube')
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in p_values])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Line chart: Messages in transit per step (for smallest p)
    ax = axes[1]
    p = p_values[0]
    ring_transit = results[p]['ring'][2]
    hyper_transit = results[p]['hypercube'][2]
    
    ax.plot(range(1, len(ring_transit)+1), ring_transit, 'o-', label='Ring', color='#f59e0b')
    if results[p]['mesh']:
        mesh_transit = results[p]['mesh'][2]
        ax.plot(range(1, len(mesh_transit)+1), mesh_transit, 's-', label='Mesh', color='#10b981')
    ax.plot(range(1, len(hyper_transit)+1), hyper_transit, '^-', label='Hypercube', color='#6366f1')
    
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Messages Not Yet Delivered')
    ax.set_title(f'Messages in Transit per Step (p={p})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topology_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: topology_comparison.png")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Topology-Aware All-to-All Personalized Communication")
    print("=" * 60)
    
    # Part A: Visualize topologies
    p = 16
    d = int(np.log2(p))
    
    print("\n--- Part A: Topology Visualization ---")
    ring_nbrs = build_ring(p)
    mesh_nbrs = build_mesh(p)
    hyper_nbrs = build_hypercube(d)
    
    fig1, _ = visualize_topology(ring_nbrs, p, "Ring")
    plt.savefig('ring_topology.png', dpi=150, bbox_inches='tight')
    
    fig2, _ = visualize_topology(mesh_nbrs, p, "2D Mesh")
    plt.savefig('mesh_topology.png', dpi=150, bbox_inches='tight')
    
    fig3, _ = visualize_topology(hyper_nbrs, p, "Hypercube")
    plt.savefig('hypercube_topology.png', dpi=150, bbox_inches='tight')
    
    print("Topology visualizations saved.")
    
    # Part B & C: Run algorithms and compare
    print("\n--- Part B & C: Algorithm Execution & Comparison ---")
    
    # Use perfect squares that are also powers of 2 for fair comparison
    results = run_comparison(p_values=[16])
    
    # Extended comparison
    print("\n--- Extended Comparison ---")
    results_ext = run_comparison(p_values=[4, 16, 64])
    
    # Plot results
    plot_comparison(results_ext)
    
    # Summary Table
    print("\n" + "=" * 70)
    print(f"{'TOPOLOGY':<12} {'STEPS':<10} {'COMPLEXITY':<15} {'WHY?'}")
    print("=" * 70)
    print(f"{'Ring':<12} {'p-1':<10} {'Linear':<15} Simple but slow for many nodes.")
    print(f"{'Mesh':<12} {'2(sqrt(p)-1)':<10} {'Square root':<15} Better; splits work into rows/columns.")
    print(f"{'Hypercube':<12} {'log2(p)':<10} {'Logarithmic':<15} Fastest steps, but high traffic on links.")
    print("=" * 70)
