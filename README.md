# S<sup>2</sup>GPT-PINN：Sparse and Small models for PDEs
We propose S<sup>2</sup>GPT-PINN, a sparse and small model for solving parametric partial differential equations (PDEs). Similar to Small Language Models (SLMs), S<sup>2</sup>GPT-PINN is tailored to domain-specific (families of) PDEs and characterized by its compact architecture and minimal computational power. Leveraging a small amount of extremely high quality data via a mathematically rigorous greedy algorithm that is enabled by the large full-order models, S<sup>2</sup>GPT-PINN relies on orders of magnitude less parameters than PINNs to achieve extremely high efficiency via two levels of customizations. The first is knowledge distillation via task-specific activation functions that are transferred from Pre-Trained PINNs. The second is a judicious down-sampling when calculating the physics-informed loss of the network compressing the number of data sites by orders of magnitude to the size of the small model.


# S<sup>2</sup>GPT-PINN Architecture:
![image](https://github.com/DuktigYajie/S2GPT-PINN/blob/main/S2GPT-PINN-Schematic.png)
