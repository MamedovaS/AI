import numpy as np

def evolutionary_strategies(u, N_max, epsilon):
    data = np.loadtxt(r"C:\Users\S_N_V\Downloads\ES_data_3.dat")  # Load data from file
    input_data = data[:, 0]  # Assign X values to input array
    output_data = data[:, 1]  # Assign Y values to output array

    tau1 = 1 / np.sqrt(2 * 3)
    tau2 = 1 / np.sqrt(2 * np.sqrt(3))
    #population
    generation = np.zeros((u, 7))
    for i in range(u):
        a = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        c = np.random.uniform(-10, 10)
        sigma_a = np.random.uniform(0, 10)
        sigma_b = np.random.uniform(0, 10)
        sigma_c = np.random.uniform(0, 10)
        MSE = mse_value(input_data, output_data, [a, b, c])
        generation[i, :] = [a, b, c, sigma_a, sigma_b, sigma_c, MSE]

    best_individual = generation[0, :]
    #ofspring
    for _ in range(N_max):
        offsprings = np.zeros((u * 5, 7))
        for j in range(u):
            for k in range(5):
                a = generation[j, 0] + np.random.randn() * generation[j, 3]
                b = generation[j, 1] + np.random.randn() * generation[j, 4]
                c = generation[j, 2] + np.random.randn() * generation[j, 5]
                R1 = np.random.randn() * tau1
                R2 = np.random.randn() * tau2
                sigma_a = generation[j, 3] * np.exp(R1) * np.exp(R2)
                R2 = np.random.randn() * tau2
                sigma_b = generation[j, 4] * np.exp(R1) * np.exp(R2)
                R2 = np.random.randn() * tau2
                sigma_c = generation[j, 5] * np.exp(R1) * np.exp(R2)
                MSE = mse_value(input_data, output_data, [a, b, c])
                offsprings[j * 5 + k, :] = [a, b, c, sigma_a, sigma_b, sigma_c, MSE]


def mse_value(input_data, output_data, params):
    a, b, c = params
    output_approx = a * (input_data ** 2) - b * np.cos(c * np.pi * input_data)
    return np.mean((output_approx - output_data) ** 2)