import numpy as np
import matplotlib.pyplot as plt

from ex2_utils import generate_responses_1




def metropolis_local_maxima(pdf, x0, n_iter, step_size, plot = False):
    x0 = np.array(x0)
    print("x0: ",x0)
    x = np.array(x0)
    xs = []

    for _ in range(n_iter):
        proposed_point = x + np.random.normal(scale=step_size, size=x.shape)
        
        # Ensure proposed point is within bounds
        proposed_point = np.clip(proposed_point, 0, np.array(pdf.shape) - 1)
        proposed_point = np.array(list(map(int, map(round, proposed_point))))
        
        # Compute PDF values at current and proposed points
        current_pdf_value = pdf[tuple(x)]
        proposed_pdf_value = pdf[tuple(proposed_point)]

        # Accept or reject proposed point
        if proposed_pdf_value >= current_pdf_value:
            print("Proposed is better: ", proposed_point)
            x = proposed_point
            xs.append(proposed_point)
        else:
            acceptance_prob = proposed_pdf_value / current_pdf_value
            if np.random.rand() < acceptance_prob:
                print("Step back.")
                x = proposed_point
                xs.append(proposed_point)
            else:
                print("Stay.")
    
    if plot:
        plt.imshow(pdf)
        a = np.array(xs)
        plt.scatter(a[0,:],a[1,:],c="r",s=1)
        plt.show()
        

    return x

# Example usage:
# Assume `pdf` is a 2D array representing the probability density function
pdf = generate_responses_1()
x0 = [20,20]  # Initial point for the search
n_iter = 10  # Number of iterations
step_size = 10 # Step size for proposing new points

maxima = metropolis_local_maxima(pdf, x0, n_iter, step_size, plot = True)
print("Local maxima found:", maxima)
