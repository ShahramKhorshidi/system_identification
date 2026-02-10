# System_Identification
**Physically-Consistent Parameter Identification of Robots in Contact.**

This repository provides an offline system identification pipeline to estimate **physically-consistent inertial parameters** of legged robots from joint torque measurements. The main focus is to deal with contact-rich dynamics of legged robots and guarantee *physical feasibility* of the identified parameters (e.g., positive-definite rotational inertia, valid mass and COM relationships), which is essential for stable simulation, control, and downstream estimation.

The codebase includes **two complementary solvers**:

- **(1) Convex LMI-based identification (SDP)**  
  A semidefinite-programming formulation with physical-consistency constraints expressed as LMIs. Solved via **CVXPY** with the **MOSEK** backend.

- **(2) Nonlinear least-squares (NLS) with log-Cholesky parameterization**  
  A nonlinear formulation that enforces physical consistency through a **log-Cholesky parameterization** of inertia-related matrices, enabling unconstrained optimization while maintaining valid inertial quantities by construction.

---

## Key features
- Offline identification from joint torques (and corresponding trajectories)
- Enforces physically-consistent inertial parameters (no “non-physical” inertia tensors)
- Two solver backends: **LMI/SDP** and **NLS (log-Cholesky)**
- Robot model loading via URDF / Pinocchio
- Demos and example robot description files included

---

## Repository structure (high-level)
- `src/` : main identification code (dynamics, solvers)
- `utils/` : utilities (plotting, etc.)
- `demo/` : runnable example scripts
- `data/` : example datasets / logs
- `files/spot_description/` : example robot model assets (URDF/meshes/etc.)
- `environment.yml` : conda environment specification

---

## Installation

### (1) Clone
```
git clone https://github.com/ShahramKhorshidi/system_identification.git
cd system_identification
```

### (2) Create the conda environment
Using the provided environment file:
```
conda env create -f environment.yml
conda activate sys_idn
```

### (3) Installing MOSEK (required for the LMI/SDP solver)
The LMI solver relies on MOSEK as the SDP backend (via CVXPY). MOSEK requires (i) installing the Python package and (ii) installing a license.

### (i) Install the MOSEK Python package
Recommended (Conda):
```
conda install -c mosek mosek
```
### (ii) Get a license (academic or trial)
Personal Academic [license](https://www.mosek.com/products/academic-licenses/)  is free for faculty/students/staff at degree-granting institutions (request must use an academic email).

---
## Run the identification
### (1) Run a demo
Check the demo/ folder for example scripts/notebooks. Typical workflow:

- Load a robot model (URDF + meshes) via Pinocchio
- Load trajectories / torque measurements
- Run either:
    - LMI/SDP solver (physically-consistent via convex constraints)
    - NLS log-Cholesky solver (physically-consistent via parameterization)
- Export identified parameters and evaluate reconstruction error / residuals

```
python demo/run_identification.py --robot spot --solver lmi
python demo/run_identification.py --robot spot --solver nls
```

The data from Spot quadruped includes 10,500 samples of the robot (collected at 100 Hz) performing various trajectories, such as base wobbling, squatting with all feet in contact, forward-backward walking, and side-to-side walking.

### (2) Adding a New Robot
The framework is robot-agnostic and can be extended to new platforms by providing a robot model and a corresponding configuration file.

#### (i) Robot description (URDF)
To add a new robot:

- Include the robot URDF file and associated assets (meshes, materials).
- Place them under:
```
files/<robot_name>_description/
```
- The URDF must define:
  - Correct joint ordering
  - Link inertial parameters (used as nominal values)
  - Consistent link and frame naming

#### (ii) Robot configuration file
For each robot, a YAML configuration file must be provided.  
Use `spot_config.yaml` as a reference template.

The configuration file specifies:
- `name`: robot name
- `mass`: robot mass
- `link_names`: list of links included in the identification
- `end_effector_frame_names`: frames used for contact modeling

The entries in `link_names` and `end_effector_frame_names` **must exactly match the corresponding link and frame names defined in the URDF**. Any mismatch will lead to incorrect regressor construction or runtime errors.

#### (iii) Mesh files (required for LMI solver)
When using the **LMI solver**, geometric information is required to construct **bounding ellipsoids** for each link, which enforce physical consistency of the inertial parameters.

Therefore:
- Each identified link must have an associated **mesh file** (or a rough geometric approximation).
- Mesh files must:
  - Be placed in the robot description folder
  - Have the **same name as the corresponding link**
  - Represent a reasonable approximation of the link geometry

These meshes are used **only for physical consistency constraints** and do not need to be visually accurate.

#### (iv) Updating the demo script
After adding the robot description and configuration:
- Register the robot in `demo/run_identification.py`
- Include robot trajectories for identification in the data folder
- Run:
```
python demo/run_identification.py --robot <robot_name> --solver <solver_name>
```

---

## Citation

If you use this repository in an academic work, please cite the following paper:
```
@inproceedings{khorshidi25icra,
  title={Physically-Consistent Parameter Identification of Robots in Contact}, 
  author={Khorshidi, Shahram and Dawood, Murad and Nederkorn, Benno and Bennewitz, Maren and Khadiv, Majid},
  booktitle={Proc. of the IEEE International Conference on Robotics and Automation (ICRA)}, 
  year={2025}
}
```