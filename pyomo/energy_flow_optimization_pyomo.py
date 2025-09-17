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
print(f"Solving the Energy Flow Optimization problem with Pyomo (Part {args.part})")
print("===============================================================\n")

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
color_scheme = {"PV":'C2', "G":'C1', "B":'C0', "C":'C3'}
print("Plotting graphs about the input data")
fig, axs = plt.subplots(1, 2, figsize = (2*5, 5))
input_data.plot(x = "time", y = ["pv", "consumption"], ylabel = "kWh", color = [color_scheme["PV"] , color_scheme["C"]], title = "PV production & Energy consumed", ax = axs[0])
input_data.plot(x = "time", y = ["buy", "sell"], color = ["r", "g"], ylabel = "cents/kWh", title = "Buy and sell prices", ax = axs[1])
fig.suptitle("Input data")
fig.savefig(folder_plots + "input_data.svg")

# Given values
charging_efficiency = .92
battery_capacity_value = 160
max_charge_value = 100
max_discharge_value = 100
max_sell_grid_value = 700
max_buy_grid_value = 700
if args.part == "C":
    packet_size = 100
    battery_extension_amount = 100
    battery_extension_cost = 1000

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
if args.part in ["B", "C"]:
    model.to_buy = pyo.Var(model.T, domain = pyo.Binary, doc ="Decision variable on whether to buy from grid or not")
    model.to_sell = pyo.Var(model.T, domain = pyo.Binary, doc ="Decision variable on whether to sell from grid or not")
if args.part == "C":
    model.gc_n = pyo.Var(model.T, domain = pyo.NonNegativeIntegers, doc = "Number of packets for Grid to consumer")
    model.bg_n = pyo.Var(model.T, domain = pyo.NonNegativeIntegers, doc = "Number of packets for Battery to grid")
    model.gb_n = pyo.Var(model.T, domain = pyo.NonNegativeIntegers, doc = "Number of packets for Grid to battery")
    model.pvg_n = pyo.Var(model.T, domain = pyo.NonNegativeIntegers, doc = "Number of packets for Photovoltaic to grid")
    model.add_battery = pyo.Var(model.T, domain = pyo.Binary, doc = "Add capacity or not")
    model.battery_capacity = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "Capacity of the battery")
    
# Parameters
print("Creating parameters")
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

def Cobj_battery(model):
    battery_addition = sum(model.add_battery[t] for t in model.T)*battery_extension_cost
    return Cobj(model) + battery_addition

if args.part == "C":
    model.obj = pyo.Objective(rule = Cobj_battery)
else:
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

if args.part == "C":
    def charge_capacity(model, t):
        # the charge capacity of the battery should not exceed a certain threshold
        return model.charge_level[t] <= model.battery_capacity[t]
else:
    def charge_capacity(model, t):
        # the charge capacity of the battery should not exceed a certain threshold
        return model.charge_level[t] <= battery_capacity_value
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

if args.part in ["B", "C"]:
    def buy_or_sell(model, t):
        # We either buy or sell: only one of the two variables can be 1
        return model.to_buy[t] + model.to_sell[t] <= 1
    model.buy_or_sell = pyo.Constraint(model.T, rule = buy_or_sell)

    def do_buy(model, t):
        # If to_buy == 0 we do not buy; we use the Big-M method
        M = 1e5
        eb_t = model.gb[t] +  model.gc[t] # Energy bought from grid
        return eb_t <= model.to_buy[t]*M
    model.do_buy = pyo.Constraint(model.T, rule = do_buy)

    def do_sell(model, t):
        # If to_sell == 0 we do not sell; we use the Big-M method
        M = 1e5
        es_t = model.bg[t] +  model.pvg[t] # Energy sold to grid
        return es_t <= model.to_sell[t]*M
    model.do_sell = pyo.Constraint(model.T, rule = do_sell)
    
if args.part == "C":
    def gc_packet(model, t):
        # GC should be bought in packets
        return model.gc[t] == model.gc_n[t]*packet_size
    model.gc_packet = pyo.Constraint(model.T, rule = gc_packet)
    
    def bg_packet(model, t):
        # BG should be sold in packets
        return model.bg[t] == model.bg_n[t]*packet_size
    model.bg_packet = pyo.Constraint(model.T, rule = bg_packet)
    
    def gb_packet(model, t):
        # GB should be bought in packets
        return model.gb[t] == model.gb_n[t]*packet_size
    model.gb_packet = pyo.Constraint(model.T, rule = gb_packet)
    
    def pvg_packet(model, t):
        # PVG should be sold in packets
        return model.pvg[t] == model.pvg_n[t]*packet_size
    model.pvg_packet = pyo.Constraint(model.T, rule = pvg_packet)
    
    def battery_capacity_increase(model, t):
        # The charge capacity of the battery
        if t == 0:
            return model.battery_capacity[t] == battery_capacity_value + model.add_battery[t]*battery_extension_amount
        else:
            return model.battery_capacity[t] == model.battery_capacity[t - 1] + model.add_battery[t]*battery_extension_amount
    model.battery_capacity_increase = pyo.Constraint(model.T, rule = battery_capacity_increase)
    
# Create a solver and launch it
if args.part == "C":
    SOLVER = "scip"
else:
    SOLVER = "glpk"
print(f"Solving with {SOLVER}")
opt = pyo.SolverFactory(SOLVER)
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
if args.part in ["A", "B"]:
    capacity_list = [battery_capacity_value for t in model.T]
else:
    capacity_list = [pyo.value(model.battery_capacity[t]) for t in model.T]
if args.part in ["B", "C"]:
    to_buy_list = [pyo.value(model.to_buy[t]) for t in model.T]
    to_sell_list = [pyo.value(model.to_sell[t]) for t in model.T]


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
pretitle = f"Part {args.part} with Pyomo ({SOLVER}) - "

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