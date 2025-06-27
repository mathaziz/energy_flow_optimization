# Energy Flow Optimization

This repository provides an example of Energy Flow Optimization, where we optimize energy flows in a system consisting of a 1) photovoltaic (PV) system, 2) an electrical battery, 3) a connection to the external electrical grid and 4) a consumer. The objective is to meet the predicted electrical energy consumption while minimizing costs.

## Dependencies

The implementation is done in Python 3 and uses the following libraries: pandas, pyomo, numpy, matplotlib, and glpk. It is advised to use Conda to manage the different packages and their version dependencies.

1. **Install Conda**:
   - Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). Miniconda is sufficient, and you do not need to register if you get it from this [link](https://repo.anaconda.com/miniconda/).

2. **Create and Activate Environment**:

   ```bash
   conda create --name enflo
   conda activate enflo
   ```
Here enflo is the name of the environment, feel free to adapt it.

3. **Install Python Packages**:

   ```bash
   conda install numpy matplotlib pandas conda-forge::pyomo conda-forge::glpk
   ```

## Usage

To run the program, follow these steps:

1. **Navigate to the Project Directory**: Open a terminal and navigate to the folder containing the project.
   ```bash
   cd path_to_project/partA/
   ```
Replace `path_to_project` with the actual path to your project directory.

2. **Check Input Files**: Ensure the folder contains an adequate `test_data.xlsx` file. A copy of the file can be found in the root directory.

3. **Execute the Program**: Run the following command:
   ```bash
   python energy_flow_optimization_partA.py
   ```

4. **Review Outputs**: Ensure that the program was executed correctly by checking the output message generated and the plots saved into the `plots\` folder.

## Mathematical model

Please read the file *math_model.pdf* for information about the proposed mathematical model.

## Author & Contact

**Amine Abdellaziz**

For any questions or feedback, please feel free to reach out:

- Email: [abdellazizamine@posteo.net](mailto:abdellazizamine@posteo.net)
- GitHub: [@mathaziz](https://github.com/mathaziz)
