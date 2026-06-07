# PDC Concepts Implementation

> Parallel & Distributed Computing — Core Concepts Implemented in Python

This repository contains three standalone Python implementations of fundamental **Parallel & Distributed Computing (PDC)** concepts, covering topology-aware communication, circular shifting on mesh networks, and hypercube dimension-ordered routing.

---

## 📁 Files & Concepts

### 1. `topology_aware_alltoall_communication.py`

**Concept:** Topology-Aware All-to-All Personalized Communication

- Builds three network topologies: **Ring**, **2D Mesh**, and **Hypercube**
- Implements all-to-all personalized communication on each topology
- Compares step complexity:
  - Ring: `O(p − 1)` — linear
  - Mesh: `O(2(√p − 1))` — square root
  - Hypercube: `O(log₂ p)` — logarithmic
- Generates topology visualizations and performance comparison charts

---

### 2. `mesh_circular_shift_double_buffering.py`

**Concept:** Circular Shift on a 2D Toroidal Mesh with Double Buffering

- Performs a K-position circular shift on an R×C toroidal (wrap-around) mesh
- Decomposes shift into three phases: **row shift → column shift → carry adjustment**
- Uses **double buffering** to prevent data overwrites during in-place shifting
- Employs **barrier synchronization** across multi-threaded workers
- Includes correctness verification and performance analysis

---

### 3. `hypercube_alltoall_dimension_routing.py`

**Concept:** All-to-All Personalized Communication on a d-Dimensional Hypercube

- Implements **d-step dimension-ordered routing** where each step resolves one bit of the destination address
- Uses **double buffering** and **barrier synchronization** for thread-safe message exchange
- Provides both sequential simulation and multi-threaded implementation
- Analyzes message complexity: `O(p − 1)` messages per node, `O(p(p − 1))` globally
- Critical path: `O(log₂ p)` steps — adding one dimension doubles nodes but only adds one step

---

## 🚀 How to Run

Each file is self-contained and can be run independently:

```bash
python topology_aware_alltoall_communication.py
python mesh_circular_shift_double_buffering.py
python hypercube_alltoall_dimension_routing.py
```

### Requirements

- Python 3.7+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## 📊 Generated Visualizations

| File | Description |
|------|-------------|
| `ring_topology.png` | Ring topology node layout |
| `mesh_topology.png` | 2D Mesh topology grid layout |
| `hypercube_topology.png` | Hypercube topology node layout |
| `topology_comparison.png` | Steps & transit comparison across all three topologies |
| `mesh_shift_performance.png` | Circular shift time vs K and grid size |
| `hypercube_performance.png` | Hypercube execution time vs dimension |

---

## 📝 Key PDC Concepts Demonstrated

- **Network Topologies** — Ring, 2D Mesh (Toroidal), Hypercube
- **All-to-All Personalized Communication** — Every node sends unique messages to every other node
- **Circular Shift** — Data rotation across a distributed grid
- **Double Buffering** — Read from current buffer, write to next buffer to avoid race conditions
- **Barrier Synchronization** — All threads must reach the barrier before any can proceed
- **Dimension-Ordered Routing** — Route messages by resolving one address bit per step
- **Multi-threaded Simulation** — Python `threading` with locks and barriers

---

## 📄 License

This project is for educational purposes — PDC coursework implementations.
