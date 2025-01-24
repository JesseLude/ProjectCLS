import matplotlib.pyplot as plt
import porespy as ps
import numpy as np

# Generate binary slices with circles
binary_slices_with_circles = generate_binary_slices_with_circles(cropped_A, axis='z', circle_radius=2)

# Analyze each slice for fractal dimension
for i, binary_slice in enumerate(binary_slices_with_circles[:5]):  # Analyze the first 5 slices
    print(f"Analyzing slice {i + 1}")
    
    # Use the boxcount method
    data = ps.metrics.boxcount(binary_slice)
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('box edge length')
    ax1.set_ylabel('number of boxes spanning phases')
    ax2.set_xlabel('box edge length')
    ax2.set_ylabel('slope')
    ax2.set_xscale('log')
    ax1.plot(data.size, data.count, '-o', label=f'Slice {i + 1}')
    ax2.plot(data.size, data.slope, '-o', label=f'Slice {i + 1}')
    
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()

