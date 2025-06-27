"""
Energy Flow Optimization Model

Optimizes energy flows in a system with PV, battery, and grid to minimize costs.

Outputs:
- DataFrame with optimized energy flows and battery charge levels.
- Plots of energy flows, battery charge levels and consumption over time.

Dependencies:
- pandas, amplpy, numpy, matplotlib

Author: Amine Abdellaziz
Date: 2025-06-27
"""

import pandas as pd
from amplpy import ampl_notebook
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

# Preparing the solver
SOLVER = "scip"
ampl = ampl_notebook(modules = [SOLVER], license_uuid="0fe956cf-71b4-4d21-9bcf-e56274e4232b")
ampl.read("PartA_ampl.model")

# Indices
print(f"Data index : {input_data.index}")
ampl.set["T"] = input_data.index

# Parameters
ampl.param["conso"] = input_data['consumption'] # Needs of the consumer"
ampl.param["lcos"] = input_data['lcos']         # LCOS
ampl.param["sell"] = input_data['sell']         # Sell prices
ampl.param["buy"] = input_data['buy']           # "Buy prices
ampl.param["pv"] = input_data['pv']             # PV output
ampl.param["charging_efficiency"] = .92
ampl.param["charge_capacity_value"] = 160
ampl.param["max_charge_value"] = 100
ampl.param["max_discharge_value"] = 100
ampl.param["max_sell_grid_value"] = 700
ampl.param["max_buy_grid_value"] = 700

# set solver und solve
ampl.solve(solver=SOLVER)

# Print results
print(f"Solve status: {ampl.solve_result}")
optimized_cost = ampl.obj["cost"].value()
print(f"Optimized cost (value of the objective function): {.01*optimized_cost:.2f} euros")

# Saving results into lists
buffer = ampl.var["gb"].to_list()
gb_list = [b for (a, b) in buffer]

buffer = ampl.var["bg"].to_list()
bg_list = [b for (a, b) in buffer]

buffer = ampl.var["bc"].to_list()
bc_list = [b for (a, b) in buffer]

buffer = ampl.var["gc"].to_list()
gc_list = [b for (a, b) in buffer]

buffer = ampl.var["pvg"].to_list()
pvg_list = [b for (a, b) in buffer]

buffer = ampl.var["pvc"].to_list()
pvc_list = [b for (a, b) in buffer]

buffer = ampl.var["pvb"].to_list()
pvb_list = [b for (a, b) in buffer]

buffer = ampl.var["charge_level"].to_list()
charge_list = [b for (a, b) in buffer]


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
sell = np.array(pvg_list) + np.array(bg_list)
buy = np.array(gb_list) + np.array(gc_list) 
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