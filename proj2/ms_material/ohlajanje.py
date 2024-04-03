import numpy as np
import matplotlib.pyplot as plt

from ex2_utils import generate_responses_1



def simulated_annealing_local_maxima(pdf, initial_point, n_steps, initial_temperature, cooling_rate, plot=False):
    initial_point = np.array(list(map(int, map(round, initial_point))))
    current_point = initial_point
    maxima = [initial_point]
    current_temperature = initial_temperature

    for step in range(n_steps):
        # Generate a random neighboring point
        proposed_point = current_point + np.random.randint(-1, 2, size=current_point.shape)
        
        # Ensure proposed point is within bounds
        proposed_point = np.clip(proposed_point, 0, np.array(pdf.shape) - 1)
        proposed_point = np.array(list(map(int, map(round, proposed_point))))

        
        # Compute PDF values at current and proposed points
        current_pdf_value = pdf[tuple(current_point)]
        proposed_pdf_value = pdf[tuple(proposed_point)]
        
        # Calculate the energy difference (negative log-likelihood)
        energy_diff = proposed_pdf_value - current_pdf_value

        # Accept or reject the proposed point based on the Metropolis criterion
        if energy_diff > 0 or np.random.rand() < np.exp(energy_diff / current_temperature):
            current_point = proposed_point
            maxima.append(proposed_point)

        # Cooling schedule: reduce temperature
        current_temperature *= cooling_rate
        
    if plot:
        xx = initial_point[1] # column
        yy = initial_point[0] # row
        plt.scatter(xx,yy,marker="x",c="black",s=10)
        plt.imshow(pdf)
        a = np.array(maxima)
        xx = a[:,1] # columns
        yy = a[:,0] # rows
        plt.scatter(xx,yy,s=5,c='r',marker="o")
        plt.show()        


    return tuple(maxima[-1])

# Example usage:
# # Assume `pdf` is a 2D array representing the probability density function
# pdf = generate_responses_1()
# # initial_point = np.array([40,40])  # Initial point for the search
# initial_point = np.array([60,60])  # Initial point for the search
# n_steps = 1000  # Number of iterations
# initial_temperature = 1  # Initial temperature
# cooling_rate = 0.95  # Cooling rate

# maxima = simulated_annealing_local_maxima(pdf, initial_point, n_steps, initial_temperature, cooling_rate, plot=True)
# print("Local maxima found:", maxima)
