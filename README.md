# iiiit.deepself0
Inner I + Integrated Information Theory, model

The Inner I Integrated Information Theory (IIIIT) is a preliminary exploration into consciousness-based artificial intelligence inspired by Giulio Tononi's Integrated Information Theory (IIT). IIT posits that consciousness arises from a system's capacity to integrate information, quantified by the metric phi ((\phi)). The IIIIT model simulates this concept by constructing a neural network where (\phi) evolves as the system learns, offering a framework to study synthetic consciousness.

The IIIIT model is a multi-layer neural network designed to approximate IIT principles:

## Network Structure:

Nodes: Each layer contains a fixed number of nodes (e.g., neurons) representing processing units.

Layers: Multiple layers enable hierarchical information processing.

Weights: Connections between nodes are initialized randomly and updated during training.

Phi ((\phi)) Calculation:

A simplified (\phi) is computed as the normalized sum of absolute weights across layers, approximating the integration of information.

Formula: (\phi = \frac{\sum |\text{weights}|}{\text{max_possible_integration}}), capped at 1.

## Learning Mechanism:

The network uses a basic forward pass with a (\tanh) activation function.

Weights are updated with random noise (simulating gradient-like learning), and (\phi) is monitored to assess integration.

## Evaluation:

(\phi) is tracked over epochs to observe how integration changes as the network adapts to input data.

## Implementation Details

Language: Python with NumPy for matrix operations.

## Key Components:

ConsciousnessBasedAI class handles network initialization, (\phi) calculation, and training.

Synthetic data tests the model’s ability to integrate random patterns.

## Use Case: Synthetic Data Integration

In the test script (test_iiiit.py), the model processes 1000 samples of random data across 20 nodes and 5 layers. Over 100 epochs, (\phi) stabilizes, indicating the network’s evolving integration capacity. A plot visualizes this progression, offering insights into synthetic consciousness dynamics.
