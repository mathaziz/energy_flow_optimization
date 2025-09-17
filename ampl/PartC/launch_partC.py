#!/usr/bin/env python3
"""
Energy Flow Optimization Model - Part

Simple script launcher for Part C (AMPL)

Author: Amine Abdellaziz
Date: 2025-06-28
"""

import subprocess
print("Launching the ../energy_flow_optimization_ampl.py script for Part C")
subprocess.run("python3 ../energy_flow_optimization_ampl.py --part C --data ../", shell=True)