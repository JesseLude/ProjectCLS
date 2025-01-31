# Simulating Coral Growth in 3D

Group 4:
Shriya Bang, Yaqi Duan, Jesse Nicolaï, Dion Hoogewoonink
University of Amsterdam

## Project Overview
This project simulates the 3D growth of corals using the **Diffusion-Limited Aggregation (DLA)** model. The simulation aims to capture realistic coral growth patterns by modeling particle diffusion and attachment in three-dimensional space. A 2D version was also implemented to gain insights into the DLA process before extending it to 3D.

**Requirements**
To run our code, the following libraries are needed:

- numpy
- matplotlib
- numba
- porespy
- scipy.stats
- seaborn

**How to run**
Our main file 3d_DLA.py should be run using pyhton3. This file shows a 3d-simulation of the growth of a coral. It then makes 2d slices of the 3d simulation that will be analyzed using
the box counting method.

We also made two files for a 2d-simulation, called oop_approach.py and 2d_DLA.py. Thes files gave us a good insight on how DLA works and how we could use it in a 3d simulation.

The file waterflow.py simulates fluid dynamics and water flow interaction with the coral growth model. This might be useful for understanding real-life growth conditions.

main_project_notebook.ipynb is a Jupyter Notebook that contains data analysis and visualization, including:

- Fractal Dimension Analysis: Uses the box-counting method to compare simulation results with real coral structures.
- Growth Rate Computation: Evaluates how coral structures evolve over time.
- Surface Area to Volume Ratio Calculation: Measures complexity and branching of the coral structure.
- Statistical Tests: Includes ANOVA and Tukey’s HSD tests to compare simulated fractal dimensions with real-world datasets.

**Research Question and Hypothesis**
To what extent does the Diffusion-Limited Aggregation (DLA) model replicate coral-like structures, and how do these simulated structures compare to real corals in terms of fractal properties?

H0: The fractal dimensions of DLA-generated coral structures are not statistically comparable to those of real coral structures, indicating that the DLA model cannot effectively simulate coral-like growth patterns.

**Methods**
The methods are explained in the Notebook.

**Conclusion**
The DLA model captures some key fractal properties of real corals but has limitations. The fractal dimensions of DLA-generated corals are statistically similar to those of real corals, suggesting that the model can replicate small-scale structural complexity.
