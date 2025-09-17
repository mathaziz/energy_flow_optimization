# Energy Flow Optimization

This project solves an instance of Energy Flow Optimization using **AMPL**. It provides a model and solution to all three parts of the technical task given to me by [hymate](https://www.hymate.com/) preceding an interview. The task deals with optimizing energy flows within a system comprising a photovoltaic system, an electrical battery, a grid connection, and a consumer. The problem statement and instructions can be found in the `instructions.pdf` file.

## Dependencies

The implementation is done in Python 3 and uses the following libraries: numpy, pandas, matplotlib, and amplpy. It is advised to use Conda to manage the different packages and their version dependencies.

1. **Install Conda**:
   - Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). Miniconda is sufficient and can be downloaded from this [link](https://repo.anaconda.com/miniconda/).

2. **Create and Activate Environment**:
   ```bash
   conda create --name energy_ampl python=3.12
   conda activate energy_ampl
    ```

3. **Install Python Packages**:

   ```bash
   conda install numpy matplotlib pandas conda-forge::amplpy conda-forge::scip
    ```
4. To use AMPL, you will need a license. If you do not have one yet, please visit the [AMPL website](https://portal.ampl.com/user/ampl/request/amplce/trial) to get a Community Edition, which is sufficient to run the programs.

## Usage

To run the program, follow these steps:

1. **Check Input Files**: Ensure the root folder `fraunhofer_iee/` contains an adequate `test_data.xlsx` file. This file should include the necessary input data structured in a specific format.

2. **Navigate to the Part Directory**: Open a terminal and navigate to the folder corresponding to the part of interest. For instance, to execute Part A:

    ```bash
    cd partA/

3. **Execute the Program**: Run the following command:

    ```bash
    python3 launch_partA.py
    ```
This script launches a Python program `energy_flow_optimization.py` situated in the root folder. The program is common to all Parts and the script provides the necessay options during launch namely which Part to launch).

4. **Review Outputs**: Ensure that the program was executed correctly by checking the output message generated and the plots saved into the `plots/` folder. The program generates several plots that visualize 1) the distribution of the PV output, 2) battery charge, 3) grid contribution, and 4) the energy mix provided to the consumer.

## Mathematical Model and Optimization Solution

For detailed information about the proposed mathematical model and the solutions to each part, please refer to the `report_ampl.pdf` file included in this repository.

## License

This project is released under the MIT License. Feel free to use, modify, and distribute the code as per the license terms.

## Author & Contact

Amine Abdellaziz

For any questions or feedback, please feel free to reach out:

- Email: [abdellazizamine@posteo.net](mailto:abdellazizamine@posteo.net)
- GitHub: [@mathaziz](https://github.com/mathaziz)

## Acknowledgments

The problem statement and instructions in `instructions.pdf`, and the data in `test_data.xlsx`, are the property of [hymate](https://portal.ampl.com/user/ampl/request/amplce/trial) and are reproduced here with permission.
