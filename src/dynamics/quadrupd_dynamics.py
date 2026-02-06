import numpy as np
import pinocchio as pin
from src.dynamics.rigid_body_dynamics import RigidBodyDynamics


class QuadrupedDynamics(RigidBodyDynamics):
    def __init__(self, urdf_file, config_file):
        super(QuadrupedDynamics, self).__init__(urdf_file, config_file, floating_base=True)
    
    def _contact_block_dim(self) -> int:
        return 3  # Fx, Fy, Fz (point-contact model)

    def _active_contact_frames(self, cnt_schedule: np.ndarray) -> list[int]:
        mask = cnt_schedule.astype(bool)
        # ensure contact_schedule matches number of EEs
        if mask.size != self.nb_ee:
            raise ValueError(f"contact_schedule length {mask.size} != nb_ee {self.nb_ee}")
        cnt_schedule = cnt_schedule.astype(bool)
        return [fid for fid, active in zip(self.endeff_ids, cnt_schedule) if active]

    def _compute_J_c(self, cnt_schedule: np.ndarray) -> np.ndarray:
        """
        Stacked contact Jacobian for active feet.
        Shape: (3*m, nv), using the linear part only (rows 3:6 of the 6xnv frame Jacobian).
        Assumes point contacts (no contact moments at the frame origin).
        Requires update_fk(...) to have been called beforehand.
        """
        frames = self._active_contact_frames(cnt_schedule)
        m = len(frames)
        k = self._contact_block_dim()
        J_c = np.zeros((k * m, self.nv))
        for j, fid in enumerate(frames):
            J6 = pin.getFrameJacobian(self.rmodel,
                                      self.rdata,
                                      fid,
                                      pin.LOCAL_WORLD_ALIGNED)
            J_c[j*k:(j+1)*k, :] = J6[0:3, :]
        return J_c

    def _compute_lambda_c(self, ee_forces: np.ndarray, cnt_schedule: np.ndarray) -> np.ndarray:
        """
        Stacked force vector for active feet.
        ee_forces expected shape: (nb_ee*3,) or (nb_ee, 3).
        Returns shape: (3*m,)
        """
        f = ee_forces.reshape(-1, 3)
        if f.shape[0] != self.nb_ee:
            raise ValueError(f"ee_forces has {f.shape[0]} rows, expected nb_ee={self.nb_ee}")
        mask = cnt_schedule.astype(bool)
        if mask.size != self.nb_ee:
            raise ValueError(f"contact_schedule length {mask.size} != nb_ee {self.nb_ee}")
        return f[mask].ravel()