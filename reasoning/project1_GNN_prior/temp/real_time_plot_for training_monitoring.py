import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque

def real_time_training_monitor():
    # Initialize data storage
    epochs = deque(maxlen=100)
    losses = deque(maxlen=100)
    accuracies = deque(maxlen=100)
    
    # Setup plot
    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Initial empty plots
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Loss')
    line2, = ax2.plot([], [], 'r-', linewidth=2, label='Accuracy')
    
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax1.legend()
    ax2.legend()
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Training simulation
    for epoch in range(100):
        # Simulate training metrics
        loss = max(0.1, 2.0 * 0.95**epoch + np.random.normal(0, 0.05))
        accuracy = min(95, 10 + 80 * (1 - 0.95**epoch) + np.random.normal(0, 1))
        
        # Update data
        epochs.append(epoch)
        losses.append(loss)
        accuracies.append(accuracy)
        
        # Update plots
        line1.set_data(list(epochs), list(losses))
        line2.set_data(list(epochs), list(accuracies))
        
        # Adjust axes
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        
        # Draw
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
        time.sleep(0.1)  # Simulate training time
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

# Run the function
real_time_training_monitor()