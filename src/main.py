from mealpy.swarm_based import WOA
import os


def fitness_function(solution):
    return (solution[0] + 2 * solution[1] - 7) ** 2 + (2 * solution[0] + solution[1] - 5) ** 2


params = {
    "epochs": [10, 20, 50, 100, 1000],
    "pop_sizes": [10, 50, 100]
}

problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [-10, ] * 2,
    "ub": [10, ] * 2,
    "minmax": "min",
    "log_to": None,
    "save_population": True,
}

if __name__ == "__main__":
    for idx_epoch, epoch in enumerate(params["epochs"]):
        for idx_pop_size, pop_size in enumerate(params["pop_sizes"]):
            model = WOA.BaseWOA(problem_dict1, epoch=epoch, pop_size=pop_size)
            best_position, best_fitness = model.solve()

            directory = f"../output/epoch={epoch}_pop_size={pop_size}"

            if not os.path.isdir(directory):
                os.makedirs(directory)

            with open(directory + "/best_solution", 'w') as file:
                file.write(f"Best solution: {best_position}, Best fitness: {best_fitness}")

            model.history.save_global_best_fitness_chart(
                filename=directory + "/chart")
