import numpy as np
import pandas as pd
from numba import njit
import os

# TODO: add argparser to input SUBMISSION_NAME directly in terminal
SUBMISSION_NAME = "starteR_optimization"

data = pd.read_csv('./data/family_data.csv', index_col='family_id')
prediction = pd.read_csv("./submissions/" + SUBMISSION_NAME + ".csv", index_col='family_id').assigned_day.values
family_size = data.n_people.values

penalties = np.asarray([
    [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ] for n in range(family_size.max() + 1)
])


cost_matrix = np.concatenate(data.n_people.apply(lambda n: np.repeat(penalties[n, 10], 100).reshape(1, 100)))


for fam in data.index:
    for choice_order, day in enumerate(data.loc[fam].drop("n_people")):
        cost_matrix[fam, day - 1] = penalties[data.loc[fam, "n_people"], choice_order]


@njit(fastmath=True)
def cost_function(prediction, family_size, cost_matrix):
    N_DAYS = cost_matrix.shape[1]
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i, (pred, n) in enumerate(zip(prediction, family_size)):
        daily_occupancy[pred - 1] += n
        penalty += cost_matrix[i, pred - 1]

    accounting_cost = 0
    n_low = 0
    n_high = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        n_high += (n > MAX_OCCUPANCY) 
        n_low += (n < MIN_OCCUPANCY)
        diff = abs(n - n_next)
        accounting_cost += max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))

    penalty += accounting_cost
    return np.asarray([penalty, n_low, n_high])

get_cost = lambda prediction: cost_function(prediction, family_size, cost_matrix)

LB_cost = int(get_cost(prediction)[0])

os.rename(f"./submissions/{SUBMISSION_NAME}.csv", f"./submissions/{LB_cost}_{SUBMISSION_NAME}.csv")
