# System_Identification
**Physically-Consistent Parameter Identification of Robots in Contact.**

This repository provides an offline system identification pipeline to estimate **physically-consistent inertial parameters** of legged robots from joint torque measurements. The main focus is to deal with contact rich legged robots and guarantee *physical feasibility* of the identified parameters (e.g., positive-definite rotational inertia, valid mass and COM relationships), which is essential for stable simulation, control, and downstream estimation.

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
conda activate system_identification
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

### (4) Run a demo
Check the demo/ folder for example scripts/notebooks. Typical workflow:

- Load a robot model (URDF + meshes) via Pinocchio
- Load trajectories / torque measurements
- Run either:
    - LMI/SDP solver (physically-consistent via convex constraints)
    - NLS log-Cholesky solver (physically-consistent via parameterization)
- Export identified parameters and evaluate reconstruction error / residuals

---

## Citation

If you use this repository in academic work, please cite the following work:
```
@inproceedings{khorshidi25icra,
  title={Physically-Consistent Parameter Identification of Robots in Contact}, 
  author={Khorshidi, Shahram and Dawood, Murad and Nederkorn, Benno and Bennewitz, Maren and Khadiv, Majid},
  booktitle={Proc. of the IEEE International Conference on Robotics and Automation (ICRA)}, 
  year={2025}
}
```