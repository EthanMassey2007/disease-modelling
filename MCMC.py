import numpy as np
import matplotlib.pyplot as plt

# Target distribution (unnormalized) — example: standard normal
def target(x):
    return np.exp(-0.5 * x**2)

# MCMC parameters
n_samples = 10000
samples = np.zeros(n_samples)
current = 0  # starting point
proposal_std = 1.0

for i in range(n_samples):
    # Propose a new point
    proposal = np.random.normal(current, proposal_std)
    
    # Acceptance ratio
    acceptance_ratio = target(proposal) / target(current)
    
    # Accept or reject
    if np.random.rand() < acceptance_ratio:
        current = proposal
    
    samples[i] = current

# Plot results
x = np.linspace(-4, 4, 1000)
plt.hist(samples, bins=50, density=True, alpha=0.6, label="MCMC Samples")
plt.plot(x, (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2), 'r-', label="True PDF")
plt.legend()
plt.show()