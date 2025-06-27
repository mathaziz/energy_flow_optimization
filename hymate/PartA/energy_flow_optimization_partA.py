"""
Energy Flow Optimization Model

Optimizes energy flows in a system with PV, battery, and grid to minimize costs.

Outputs:
- DataFrame with optimized energy flows and battery charge levels.
- Plots of energy flows, battery charge levels and consumption over time.

Dependencies:
- pandas, pyomo, numpy, matplotlib

Author: Amine Abdellaziz
Date: 2025-05-07
"""

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import subprocess

folder_plots = "plots/"
subprocess.run(f"rm -r {folder_plots} solution.csv", shell=True)
subprocess.run(f"mkdir -v {folder_plots}", shell=True)

# Reading the Data
filename = "test_data.xlsx"
print(f"Reading data from {filename}")
input_data = pd.read_excel(filename)
input_data.columns = ["time", "pv", "consumption", "lcos", "sell", "buy"] #
input_data.head()

# Plot of the input data
print("Plotting graphs about the input data")
ax = input_data.plot(x = "time", y = ["pv", "consumption"], ylabel = "kWh", title = "PV production & Energy consumed")
ax.figure.savefig(folder_plots + "input-data_energy.svg")
ax = input_data.plot(x = "time", y = ["sell", "buy"], ylabel = "cents/kWh", title = "Sell and Buy prices")
ax.figure.savefig(folder_plots + "input-data_prices.svg")

# Given values
charging_efficiency = .92
charge_capacity_value = 160
max_charge_value = 100
max_discharge_value = 100
max_sell_grid_value = 700
max_buy_grid_value = 700

# creating model
print("Creating Pyomo model")
model = pyo.ConcreteModel()

# Indices
print("Creating Pyomo indices")
model.T = pyo.Set(initialize=input_data.index)

# Variables
print("Creating variables")
model.gc = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "Grid to consumer")
model.bg = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "Battery to grid")
model.bc = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "Battery to consumer")
model.gb = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "Grid to battery")
model.pvg = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "PV to grid")
model.pvc = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "PV to consumer")
model.pvb = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "PV to battery")
model.charge_level = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "Level of charge in battery")

# Parameters
print("Creating paranmeters")
model.conso = pyo.Param(model.T, initialize = input_data['consumption'], doc = "Needs of the consumer")
model.lcos = pyo.Param(model.T, initialize = input_data['lcos'], doc = "LCOS")
model.sell = pyo.Param(model.T, initialize = input_data['sell'], doc = "Sell prices")
model.buy = pyo.Param(model.T, initialize = input_data['buy'], doc = "Buy prices")
model.pv = pyo.Param(model.T, initialize = input_data['pv'], doc = "PV output")

# Objective function
print("Creating objective function")
def Cobj(model):
    eb = sum(model.buy[t]*(model.gb[t] +  model.gc[t]) for t in model.T) # Energy bought from grid
    es = sum(model.sell[t]*(model.bg[t] + model.pvg[t]) for t in model.T) # Energy sold to grid
    ed = sum(model.lcos[t]*(model.bg[t] + model.bc[t]) for t in model.T) # Energy discharged from battery
    return  eb - es + ed

model.obj = pyo.Objective(rule = Cobj)

# Constraints 
print("Defining constraints")
def photovoltaic(model, t):
    # The output of the PV should not exceed the production
    return model.pvg[t] + model.pvc[t] + model.pvb[t] <= model.pv[t]
model.photovoltaic = pyo.Constraint(model.T, rule = photovoltaic)

def charge_evolution(model, t):
    # The level of charge of the battery
    charge_input = model.gb[t] + model.pvb[t]
    charge_output = model.bc[t] + model.bg[t]
    if t == 0:
        return model.charge_level[t] == charging_efficiency*charge_input - charge_output # We start with an empty battery
    else:
        return model.charge_level[t] == model.charge_level[t - 1] + charging_efficiency*charge_input - charge_output
model.charge_evolution = pyo.Constraint(model.T, rule = charge_evolution)

def charge_capacity(model, t):
    # the charge capacity of the battery should not exceed a certain threshold
    return model.charge_level[t] <= charge_capacity_value
model.charge_capacity = pyo.Constraint(model.T, rule = charge_capacity)

def max_charge(model, t):
    # maximum charge
    return model.gb[t] + model.pvb[t] <= max_charge_value
model.max_charge = pyo.Constraint(model.T, rule = max_charge)

def max_discharge(model, t):
    # maximum discharge
    return model.bg[t] + model.bc[t] <= max_discharge_value
model.max_discharge = pyo.Constraint(model.T, rule = max_discharge)

def max_sell_grid(model, t):
    # We cannot sell to the grid more than a certain threshold
    return model.bg[t] + model.pvg[t] <= max_sell_grid_value
model.max_sell_grid = pyo.Constraint(model.T, rule = max_sell_grid)

def max_buy_grid(model, t):
    # We cannot buy from the grid more than a certain threshold
    return model.gb[t] + model.gc[t] <= max_buy_grid_value
model.max_buy_grid = pyo.Constraint(model.T, rule = max_buy_grid)

def consumption(model, t):
    # We need to satisfy the demand of the consumer
    return model.gc[t] + model.pvc[t] + model.bc[t] == model.conso[t]
model.consumption = pyo.Constraint(model.T, rule = consumption)

# Create a solver and launch it
print("Solving with GLPK")
opt = pyo.SolverFactory('glpk')
results = opt.solve(model)

# Print results
print(f"Solve status: {results.solver.status}")
optimized_cost = pyo.value(model.obj)
print(f"Optimized cost (value of the objective function): {.01*optimized_cost:.2f} euros")

# Saving results into lists
gb_list = [pyo.value(model.gb[t]) for t in model.T]
bg_list = [pyo.value(model.bg[t]) for t in model.T]
bc_list = [pyo.value(model.bc[t]) for t in model.T]
gc_list = [pyo.value(model.gc[t]) for t in model.T]
pvg_list = [pyo.value(model.pvg[t]) for t in model.T]
pvc_list = [pyo.value(model.pvc[t]) for t in model.T]
pvb_list = [pyo.value(model.pvb[t]) for t in model.T]
charge_list = [pyo.value(model.charge_level[t]) for t in model.T]

# Creating a pandas DataFrame
filename = 'solution.csv'
print(f"Saving results into CSV file : {filename}")
output_data = pd.DataFrame({
    "time":input_data["time"],
    "pv to grid":pvg_list,
    "pv to consumer":pvc_list, 
    "pv to battery":pvb_list,
    "grid to battery":gb_list,
    "grid to consumer":gc_list, 
    "battery to grid":bg_list,
    "battery to consumer":bc_list,
    "charge":charge_list, 
})
output_data.to_csv(filename)

# Plots
print("Plotting results")

# PV 
x = input_data["time"]
y = np.vstack([pvc_list, pvg_list, pvb_list])
labels = ["Consumer", "Grid", "Battery"]
fig, ax = plt.subplots()
ax.plot(x, input_data["pv"], color = 'r', label = "PV production")
ax.stackplot(x, y, labels = labels)
ax.set_title("PV output")
ax.set_xlabel("Time")
fig.autofmt_xdate()
ax.set_ylabel("kW")
plt.legend()
fig.savefig(folder_plots + "pv_output.svg")

# Battery
ax = output_data.plot(x = "time", y = ["grid to battery", "pv to battery"], ylabel = "kW", title = "Battery input")
ax.figure.savefig(folder_plots + 'battery_input.svg')
ax = output_data.plot(x = "time", y = ["battery to grid", "battery to consumer"], ylabel = "kW", title = "Battery output")
ax.figure.savefig(folder_plots + 'battery_output.svg')
ax = output_data.plot(x = "time", y = ["charge"], ylabel = "kWh", title = "Battery charge")
ax.figure.savefig(folder_plots + 'battery_charge.svg')

# Selling and buying
x = input_data["time"]
sell = [pvg_list[t] + bg_list[t] for t in model.T]
buy = [gb_list[t] + gc_list[t] for t in model.T]
labels = ["Buy", "Sell"]
fig, ax = plt.subplots()
ax.plot(x, buy, color = 'C3', label = labels[0])
ax.plot(x, sell, color = 'C2', label = labels[1])
ax.set_title("Do we buy or do we sell?")
ax.set_xlabel("Time")
fig.autofmt_xdate()
ax.set_ylabel("kWh")
plt.legend()
fig.savefig(folder_plots + "buy_or_sell.svg")

# Grid
ax = output_data.plot(x = "time", y = ["grid to battery", "grid to consumer"], ylabel = "kW", title = "Grid output")
ax.figure.savefig(folder_plots + 'grid_output.svg')
ax = output_data.plot(x = "time", y = ["pv to grid", "battery to grid"], ylabel = "kW", title = "Grid input")
ax.figure.savefig(folder_plots + 'grid_input.svg')

# Consumer
x = input_data["time"]
y = np.vstack([pvc_list, gc_list, bc_list])
labels = ["PV", "Grid", "Battery"]
fig, ax = plt.subplots()
ax.plot(x, input_data["consumption"], color = 'r', label = "Consumption")
ax.stackplot(x, y, labels = labels)
ax.set_title(f"Consumer input - cost = {.01*optimized_cost:.2f} euros")
ax.set_xlabel("Time")
fig.autofmt_xdate()
ax.set_ylabel("kW")
plt.legend()
fig.savefig(folder_plots + "consumer_input.svg")