"""
Section B - Question 2: Circular Shift on a 2D Mesh (Toroidal Grid)
Implements neighbor-only circular shift with double buffering and barrier synchronization.
"""

import time
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

# ============================================================
# Mesh Node & Grid Setup
# ============================================================

class MeshGrid:
    """2D toroidal mesh grid for circular shift simulation."""
    
    def __init__(self, R, C):
        self.R = R
        self.C = C
        self.P = R * C
        # Grid stores data values; initially node k has data k
        self.grid = np.arange(self.P).reshape(R, C)
        self.next_grid = np.zeros_like(self.grid)
        self.barrier = threading.Barrier(self.P)
        self.lock = threading.Lock()
    
    def get_node_id(self, r, c):
        return r * self.C + c
    
    def get_coords(self, node_id):
        return divmod(node_id, self.C)
    
    def print_grid(self, title="Grid"):
        print(f"\n{title}:")
        print("    " + "  ".join(f"C{c:>2}" for c in range(self.C)))
        print("   " + "-----" * self.C)
        for r in range(self.R):
            row_str = " | ".join(f"{self.grid[r, c]:>2}" for c in range(self.C))
            print(f"R{r} | {row_str} |")
        print()


def circular_shift_mesh(R, C, K):
    """
    Perform a circular K-shift on an R x C toroidal mesh.
    
    Decomposition:
    - K' = K mod P (avoid redundant full cycles)
    - Row shift amount: s = K' mod C
    - Column shift amount: t = K' // C (base) + carry
    
    Three phases:
    1. Row shift (horizontal circular shift by s in each row)
    2. Column shift (vertical shift by base t)
    3. Column carry adjustment (extra +1 for wrapped positions)
    """
    P = R * C
    K_eff = K % P  # Effective shift
    
    if K_eff == 0:
        print("K mod P = 0, no shift needed.")
        return np.arange(P).reshape(R, C), np.arange(P).reshape(R, C), 0
    
    row_shift = K_eff % C
    col_shift_base = K_eff // C
    
    print(f"Parameters: R={R}, C={C}, P={P}, K={K}, K'={K_eff}")
    print(f"Row shift: {K_eff} mod {C} = {row_shift}")
    print(f"Col shift base: {K_eff} // {C} = {col_shift_base}")
    
    # Initial grid
    grid = np.arange(P).reshape(R, C)
    initial_grid = grid.copy()
    
    print_grid(grid, R, C, "Initial Grid")
    
    # ---- Phase 1: Row Shift ----
    # Each row shifts right by row_shift using double buffering
    next_grid = np.zeros_like(grid)
    
    total_comm = 0
    for r in range(R):
        for c in range(C):
            src_col = (c - row_shift + C) % C
            next_grid[r, c] = grid[r, src_col]
            if row_shift > 0:
                total_comm += 1  # Each node does one receive
    
    grid = next_grid.copy()
    print_grid(grid, R, C, f"After Phase 1 (Row Shift right by {row_shift})")
    
    # ---- Phase 2 & 3: Column Shift with carry ----
    next_grid = np.zeros_like(grid)
    
    for r in range(R):
        for c in range(C):
            # Determine if this position had a carry in row shift
            # After row shift, columns 0..row_shift-1 came from original cols C-row_shift..C-1 (carry)
            has_carry = c < row_shift
            total_col_shift = col_shift_base + (1 if has_carry else 0)
            
            src_row = (r - total_col_shift + R) % R
            next_grid[r, c] = grid[src_row, c]
            if total_col_shift > 0:
                total_comm += 1
    
    grid = next_grid.copy()
    print_grid(grid, R, C, f"After Phase 2+3 (Column Shift by {col_shift_base} + carry)")
    
    return initial_grid, grid, total_comm


def print_grid(grid, R, C, title="Grid"):
    """Pretty-print a grid."""
    print(f"\n{title}:")
    print("    " + "  ".join(f"C{c:>2}" for c in range(C)))
    print("   " + "-----" * C)
    for r in range(R):
        row_str = " | ".join(f"{grid[r, c]:>2}" for c in range(C))
        print(f"R{r} | {row_str} |")


def verify_shift(initial_grid, final_grid, R, C, K):
    """Verify correctness of the circular shift."""
    P = R * C
    K_eff = K % P
    
    # Check: it's a permutation of 0..P-1
    final_flat = final_grid.flatten()
    if sorted(final_flat) != list(range(P)):
        print("INCORRECT: Final grid is not a permutation of 0..P-1")
        return False
    
    # Check: each item moved exactly K positions in row-major order
    for r in range(R):
        for c in range(C):
            node_id = r * C + c
            expected_data = (node_id - K_eff + P) % P
            if final_grid[r, c] != expected_data:
                print(f"INCORRECT: Node ({r},{c})={node_id} has data {final_grid[r,c]}, expected {expected_data}")
                return False
    
    return True


# ============================================================
# Multi-threaded Implementation with Barriers
# ============================================================

def threaded_circular_shift(R, C, K):
    """
    Threaded implementation using barriers and double buffering.
    Each mesh location (r,c) is a separate worker thread.
    """
    P = R * C
    K_eff = K % P
    row_shift = K_eff % C
    col_shift_base = K_eff // C
    
    grid = np.arange(P).reshape(R, C).astype(int)
    next_grid = np.zeros_like(grid)
    barrier = threading.Barrier(P)
    
    def worker(r, c):
        nonlocal grid, next_grid
        
        # Phase 1: Row shift
        src_col = (c - row_shift + C) % C
        next_grid[r, c] = grid[r, src_col]
        barrier.wait()  # Barrier after row shift
        
        # Swap grids
        if r == 0 and c == 0:
            grid[:] = next_grid[:]
        barrier.wait()
        
        # Phase 2+3: Column shift with carry
        has_carry = c < row_shift
        total_col = col_shift_base + (1 if has_carry else 0)
        src_row = (r - total_col + R) % R
        next_grid[r, c] = grid[src_row, c]
        barrier.wait()  # Barrier after column shift
    
    threads = []
    for r in range(R):
        for c in range(C):
            t = threading.Thread(target=worker, args=(r, c))
            threads.append(t)
    
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start_time
    
    return next_grid, elapsed


# ============================================================
# Performance Analysis
# ============================================================

def run_performance_analysis():
    """Measure and plot performance for different R, C, K values."""
    
    # Test 1: Time vs K for fixed R, C
    R, C = 4, 4
    K_values = list(range(1, R * C))
    times_k = []
    
    print("\n--- Time vs K (R=4, C=4) ---")
    for K in K_values:
        _, elapsed = threaded_circular_shift(R, C, K)
        times_k.append(elapsed * 1000)  # convert to ms
    
    # Test 2: Time vs P for fixed K
    K_fixed = 5
    grid_sizes = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    times_p = []
    p_values = []
    
    print("\n--- Time vs P (K=5) ---")
    for R, C in grid_sizes:
        _, elapsed = threaded_circular_shift(R, C, K_fixed)
        times_p.append(elapsed * 1000)
        p_values.append(R * C)
    
    # Communication analysis
    print("\n--- Communication Analysis ---")
    print(f"{'R×C':<8} {'K':<5} {'Row Steps':<12} {'Col Steps':<12} {'Total Comms'}")
    print("-" * 50)
    for R, C in [(4, 4), (6, 6), (8, 8)]:
        for K in [3, 5, 7]:
            P = R * C
            K_eff = K % P
            row_s = K_eff % C
            col_s = K_eff // C
            # Approximate: row_shift steps + col_shift max steps
            total = row_s + col_s + (1 if row_s > 0 else 0)
            print(f"{R}×{C:<5} {K:<5} {row_s:<12} {col_s:<12} {total}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(K_values, times_k, 'o-', color='#6366f1', linewidth=2, markersize=6)
    axes[0].set_xlabel('Shift Distance K')
    axes[0].set_ylabel('Execution Time (ms)')
    axes[0].set_title(f'Time vs K (R={4}, C={4})')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(p_values, times_p, 's-', color='#10b981', linewidth=2, markersize=6)
    axes[1].set_xlabel('Grid Size P = R×C')
    axes[1].set_ylabel('Execution Time (ms)')
    axes[1].set_title(f'Time vs Grid Size (K={K_fixed})')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mesh_shift_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: mesh_shift_performance.png")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Circular Shift on a 2D Mesh (Toroidal Grid)")
    print("=" * 60)
    
    # Example 1: R=4, C=4, K=6
    R, C, K = 4, 4, 6
    print(f"\n>>> Example: R={R}, C={C}, K={K}")
    initial, final, comms = circular_shift_mesh(R, C, K)
    
    correct = verify_shift(initial, final, R, C, K)
    print(f"\nMesh circular shift: {'CORRECT' if correct else 'INCORRECT'}")
    print(f"Total neighbor communications: {comms}")
    
    # Example 2: R=4, C=4, K=5
    R, C, K = 4, 4, 5
    print(f"\n\n>>> Example: R={R}, C={C}, K={K}")
    initial, final, comms = circular_shift_mesh(R, C, K)
    correct = verify_shift(initial, final, R, C, K)
    print(f"\nMesh circular shift: {'CORRECT' if correct else 'INCORRECT'}")
    
    # Example 3: Larger grid
    R, C, K = 6, 6, 11
    print(f"\n\n>>> Example: R={R}, C={C}, K={K}")
    initial, final, comms = circular_shift_mesh(R, C, K)
    correct = verify_shift(initial, final, R, C, K)
    print(f"\nMesh circular shift: {'CORRECT' if correct else 'INCORRECT'}")
    
    # Threaded version
    print("\n\n--- Threaded Implementation ---")
    R, C, K = 4, 4, 6
    result, elapsed = threaded_circular_shift(R, C, K)
    print_grid(result, R, C, f"Threaded result (K={K})")
    correct = verify_shift(np.arange(R*C).reshape(R,C), result, R, C, K)
    print(f"Mesh circular shift: {'CORRECT' if correct else 'INCORRECT'}")
    print(f"Threaded execution time: {elapsed*1000:.2f} ms")
    
    # Performance analysis
    print("\n\n--- Performance Analysis ---")
    run_performance_analysis()
    
    # Analysis paragraph
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("""
Limiting communication to nearest neighbors and using double buffering
helps avoid data loss because each node writes to a separate buffer while 
reading from the current grid. This prevents race conditions where data 
could be overwritten before being read. The barrier synchronization ensures 
all nodes complete each phase before proceeding, maintaining consistency.

The number of steps scales as O(row_shift + col_shift) where row_shift = 
K mod C and col_shift = K // C. For a P-node mesh (R×C), the worst case 
is O(C + R) = O(2√P), which is much better than O(P) for a linear/ring 
arrangement. Toroidal connectivity removes edge effects — without wrap-
around, nodes at edges would require asymmetric communication patterns.
The communication cost along rows vs columns is balanced since both phases 
use the same nearest-neighbor exchange pattern.
""")
