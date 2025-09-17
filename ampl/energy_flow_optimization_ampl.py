#!/usr/bin/env python3
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
import argparse

parser = argparse.ArgumentParser(prog="energy_flow_optimization")
parser.add_argument('--part', help = 'Which part of the problem to solve (A, B or C) -- default A', default = 'A')
parser.add_argument('--data', help = 'Root directory of the data file test_data.xls -- default ./', default = './')
args = parser.parse_args()
if args.part not in ['A', 'B', 'C']:
    args.part = 'A'

print("")
print("===============================================================")
print(f"Solving the Energy Flow Optimization problem (Part {args.part})")
print("===============================================================\n")

folder_plots = "plots/"
subprocess.run(f"rm -r {folder_plots} solution.csv", shell=True)
subprocess.run(f"mkdir -v {folder_plots}", shell=True)

# Reading the Data
filename = args.data + "test_data.xlsx"
print(f"Reading data from {filename}")
input_data = pd.read_excel(filename)
input_data.columns = ["time", "pv", "consumption", "lcos", "sell", "buy"] #
input_data.head()

# Plot of the input data
color_scheme = {"PV":'C2', "G":'C1', "B":'C0', "C":'C3'}
print("Plotting graphs about the input data")
fig, axs = plt.subplots(1, 2, figsize = (2*5, 5))
input_data.plot(x = "time", y = ["pv", "consumption"], ylabel = "kWh", color = [color_scheme["PV"] , color_scheme["C"]], title = "PV production & Energy consumed", ax = axs[0])
input_data.plot(x = "time", y = ["buy", "sell"], color = ["r", "g"], ylabel = "cents/kWh", title = "Buy and sell prices", ax = axs[1])
fig.suptitle("Input data")
fig.savefig(folder_plots + "input_data.svg")

# Preparing the solver
SOLVER = "scip"
ampl = ampl_notebook(modules = [SOLVER], license_uuid="default")
if args.part == 'B':
    ampl.read("PartB_ampl.model")
elif args.part == 'C':
    ampl.read("PartC_ampl.model")
else:
    ampl.read("PartA_ampl.model")

# Indices
print(f"Data index : {input_data.index}")
ampl.set["T"] = input_data.index

# Parameters
ampl.param["conso"] = input_data['consumption'] # Needs of the consumer
ampl.param["lcos"] = input_data['lcos']         # LCOS
ampl.param["sell"] = input_data['sell']         # Sell prices
ampl.param["buy"] = input_data['buy']           # "Buy prices
ampl.param["pv"] = input_data['pv']             # PV output
ampl.param["charging_efficiency"] = .92
ampl.param["battery_capacity_value"] = 160
ampl.param["max_charge_value"] = 100
ampl.param["max_discharge_value"] = 100
ampl.param["max_sell_grid_value"] = 700
ampl.param["max_buy_grid_value"] = 700
if args.part in ['B', 'C']:
    ampl.param["big_M"] = 1e5
if args.part == 'C':
    ampl.param["packet_size"] = 100
    ampl.param["battery_extension_amount"] = 100
    ampl.param["battery_extension_cost"] = 1000
    
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

if args.part in ['B', 'C']:
    buffer = ampl.var["to_buy"].to_list()
    to_buy_list = [b for (a, b) in buffer]
    
    buffer = ampl.var["to_sell"].to_list()
    to_sell_list = [b for (a, b) in buffer]
    
if args.part == 'C':
    buffer = ampl.var["battery_capacity"].to_list()
    capacity_list = [b for (a, b) in buffer]    
if args.part in ['A', 'B']:
    capacity_list = [ampl.param["battery_capacity_value"].value() for t in charge_list]
    
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
    "battery capacity":capacity_list,
    "battery to grid":bg_list,
    "battery to consumer":bc_list,
    "charge":charge_list,
})
if args.part in ['B', 'C']:
    output_data["to_buy"] = to_buy_list
    output_data["to_sell"]= to_sell_list
output_data.to_csv(filename)

# Plots
print("Plotting results")
pretitle = f"Part {args.part} with AMPL ({SOLVER}) - "

# PV 
#----
x = input_data["time"]
y = np.vstack([pvc_list, pvg_list, pvb_list])
labels = ["Consumer", "Grid", "Battery"]
colors = [color_scheme["C"], color_scheme["G"], color_scheme["B"]]
fig, ax = plt.subplots()
ax.plot(x, input_data["pv"], color = color_scheme["PV"], label = "PV production")
ax.stackplot(x, y, labels = labels, colors = colors)
ax.set_title(pretitle + "PV output")
ax.set_xlabel("Time")
fig.autofmt_xdate()
ax.set_ylabel("kW")
plt.legend()
fig.savefig(folder_plots + "pv_output.svg")

# Battery
#--------
x = input_data["time"]
fig, axs = plt.subplots(1, 2, figsize = (2*5, 5))
# Charge level
axs[0].plot(x, capacity_list, color = 'r', ls = ":", label = "Capacity")
axs[0].stackplot(x, charge_list, color = color_scheme["B"])
axs[0].set_title("Charge level")
axs[0].set_ylabel("kWh")
axs[0].set_xlabel("Time")
axs[0].legend()
# Input $ output
axs[1].plot(x, gb_list, color = color_scheme["G"], label = "from Grid")
axs[1].plot(x, pvb_list, color = color_scheme["PV"], label = "from PV")
axs[1].plot(x, - np.array(bg_list), color = color_scheme["G"], ls = "--", label = "to Grid")
axs[1].plot(x, - np.array(bc_list), color = color_scheme["C"], ls = "--", label = "to Consumer")
axs[1].set_title("Input & Output")
axs[1].set_ylabel("kW")
axs[1].set_xlabel("Time")
axs[1].legend()
fig.autofmt_xdate()
fig.suptitle(pretitle + "Battery")
fig.savefig(folder_plots + "battery.svg")


# Grid
#-----
x = input_data["time"]
fig, axs = plt.subplots(1, 2, figsize = (2*5, 5))
# Input & output
axs[0].plot(x, -np.array(gb_list), color = color_scheme["B"], ls = "--", label = "to Battery")
axs[0].plot(x, -np.array(gc_list), color = color_scheme["C"], ls = "--", label = "to Consumer")
axs[0].plot(x, np.array(pvg_list), color = color_scheme["PV"], label = "from PV")
axs[0].plot(x, np.array(bg_list), color = color_scheme["B"], label = "from Battery")
axs[0].set_title("Input & output")
axs[0].set_ylabel("kW")
axs[0].set_xlabel("Time")
axs[0].legend()
# Selling and buying
sell = np.array(pvg_list) + np.array(bg_list)
buy = np.array(gb_list) + np.array(gc_list) 
labels = ["Buy", "Sell"]
axs[1].plot(x, buy, color = 'r', label = labels[0])
axs[1].plot(x, sell, color = 'g', label = labels[1])
axs[1].set_title("Do we buy or do we sell?")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("kW")
axs[1].legend()
fig.autofmt_xdate()
fig.suptitle(pretitle + "Grid")
fig.savefig(folder_plots + 'grid.svg')

# Consumer
#---------
x = input_data["time"]
y = np.vstack([pvc_list, gc_list, bc_list])
labels = ["PV", "Grid", "Battery"]
colors = [color_scheme["PV"], color_scheme["G"], color_scheme["B"]]
fig, ax = plt.subplots()
ax.plot(x, input_data["consumption"], color = color_scheme["C"], label = "Consumption")
ax.stackplot(x, y, labels = labels, colors = colors)
ax.set_title(pretitle + f"Consumer input - cost = {.01*optimized_cost:.2f} euros")
ax.set_xlabel("Time")
fig.autofmt_xdate()
ax.set_ylabel("kW")
plt.legend()
fig.savefig(folder_plots + "consumer_input.svg")