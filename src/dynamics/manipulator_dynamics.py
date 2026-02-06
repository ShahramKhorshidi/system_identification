import numpy as np
import pinocchio as pin
from src.dynamics.rigid_body_dynamics import RigidBodyDynamics


class ManipulatorDynamics(RigidBodyDynamics):
    def __init__(self, urdf_file, config_file):
        super(ManipulatorDynamics, self).__init__(urdf_file, config_file, floating_base=False)

    def _contact_block_dim(self) -> int:
        pass

    def _active_contact_frames(self, cnt_schedule: np.ndarray) -> list[int]:
        pass

    def _compute_J_c(self, cnt_schedule: np.ndarray) -> np.ndarray:
        pass

    def _compute_lambda_c(self, ee_forces: np.ndarray, cnt_schedule: np.ndarray) -> np.ndarray:
        pass