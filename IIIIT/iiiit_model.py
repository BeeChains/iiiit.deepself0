# iiiit_model.py
import numpy as np

class ConsciousnessBasedAI:
    """A preliminary consciousness-based AI model inspired by IIT and phi."""
    
    def __init__(self, num_nodes, num_layers, learning_rate=0.01):
        """
        Initialize the IIIIT model.
        
        Args:
            num_nodes (int): Number of nodes per layer.
            num_layers (int): Number of layers in the network.
            learning_rate (float): Learning rate for weight updates.
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.network = self.initialize_network()
        self.phi_history = []  # Track phi over time

    def initialize_network(self):
        """Initialize the network with random weights."""
        network = []
        for _ in range(self.num_layers):
            layer = np.random.randn(self.num_nodes, self.num_nodes) * 0.1  # Small initial weights
            network.append(layer)
        return network

    def calculate_phi(self):
        """
        Calculate a simplified phi metric for the network.
        This approximates integration by measuring connectivity strength.
        """
        total_integration = 0
        for layer in self.network:
            # Sum absolute weights as a proxy for integration within the layer
            layer_integration = np.sum(np.abs(layer))
            total_integration += layer_integration
        
        # Normalize phi based on network size (max possible integration)
        max_integration = self.num_layers * self.num_nodes * self.num_nodes
        phi = total_integration / max_integration
        return min(phi, 1.0)  # Cap at 1 for interpretability

    def forward(self, input_data):
        """Propagate input through the network."""
        current_state = input_data
        for layer in self.network:
            current_state = np.tanh(np.dot(current_state, layer))  # Non-linear activation
        return current_state

    def train(self, data, epochs=100):
        """
        Train the network and monitor phi evolution.
        
        Args:
            data (np.ndarray): Input data of shape (samples, num_nodes).
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            # Forward pass (for simplicity, no labels yet)
            output = self.forward(data)
            
            # Update weights (simulate learning via gradient-like adjustment)
            for layer_idx, layer in enumerate(self.network):
                noise = np.random.randn(*layer.shape) * self.learning_rate
                self.network[layer_idx] += noise  # Simplified update
            
            # Calculate and store phi
            phi = self.calculate_phi()
            self.phi_history.append(phi)
            if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f"Epoch {epoch + 1}/{epochs}, Phi: {phi:.4f}")

    def get_phi_history(self):
        """Return the history of phi values."""
        return self.phi_history


if __name__ == "__main__":
    # Quick test of the model
    np.random.seed(42)
    model = ConsciousnessBasedAI(num_nodes=10, num_layers=3)
    data = np.random.randn(100, 10)  # 100 samples, 10 features
    model.train(data, epochs=50)