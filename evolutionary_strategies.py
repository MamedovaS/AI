import numpy as np
import matplotlib.pyplot as plt


def evolutionary_strategies(u, N_max, epsilon):
    data = np.loadtxt(r"C:\Users\S_N_V\Downloads\ES_data_3.dat")  # Load data from file
    input_data = data[:, 0]  # Assign X values to input array
    output_data = data[:, 1]  # Assign Y values to output array

    tau1 = 1 / np.sqrt(2 * 3)
    tau2 = 1 / np.sqrt(2 * np.sqrt(3))

    # Create the initial population
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

        generation = generation[generation[:, 6].argsort()]  # Sort parents by MSE
        offsprings = offsprings[offsprings[:, 6].argsort()]  # Sort offsprings by MSE

        if abs(generation[0, 6] - offsprings[0, 6]) < epsilon:
            if generation[0, 6] > offsprings[0, 6]:
                best_individual = offsprings[0, :]
            break

        generation = offsprings[:u, :]
        best_individual = generation[0, :]

    MSE = mse_value(input_data, output_data, best_individual[:3])

    output_approximated = best_individual[0] * (input_data ** 2) - best_individual[1] * np.cos(
        best_individual[2] * np.pi * input_data)

    plt.figure()
    plt.plot(input_data, output_data, label='Original plot')
    plt.plot(input_data, output_approximated, 'r', label='Approximated plot')
    plt.legend()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(input_data, output_data)
    plt.title('Original plot')
    plt.subplot(1, 2, 2)
    plt.plot(input_data, output_approximated, 'r')
    plt.title('Approximated plot')
    plt.show()

    solution = [best_individual[0], best_individual[1], best_individual[2], MSE]
    return solution


def mse_value(input_data, output_data, params):
    a, b, c = params
    output_approx = a * (input_data ** 2) - b * np.cos(c * np.pi * input_data)
    return np.mean((output_approx - output_data) ** 2)


# Example usage
u = 10
N_max = 100
epsilon = 1e-6
solution = evolutionary_strategies(u, N_max, epsilon)
print("Solution:", solution)
