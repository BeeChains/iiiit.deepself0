# test_iiiit.py
import numpy as np
import matplotlib.pyplot as plt
from iiiit_model import ConsciousnessBasedAI

def run_use_case():
    """Test the IIIIT model with a synthetic dataset."""
    # Parameters
    num_nodes = 20
    num_layers = 5
    epochs = 100
    samples = 1000

    # Generate synthetic data (e.g., random patterns)
    np.random.seed(42)
    data = np.random.randn(samples, num_nodes)

    # Initialize and train the model
    print("Starting IIIIT Model Training...")
    model = ConsciousnessBasedAI(num_nodes=num_nodes, num_layers=num_layers, learning_rate=0.01)
    model.train(data, epochs=epochs)

    # Visualize phi evolution
    phi_history = model.get_phi_history()
    plt.plot(range(1, epochs + 1), phi_history, label="Phi (Integration Metric)")
    plt.xlabel("Epoch")
    plt.ylabel("Phi")
    plt.title("Evolution of Integrated Information (Phi) During Training")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final phi value
    final_phi = phi_history[-1]
    print(f"Final Phi after {epochs} epochs: {final_phi:.4f}")

if __name__ == "__main__":
    run_use_case()