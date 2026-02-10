import numpy as np
import pinocchio as pin
from src.dynamics.rigid_body_dynamics import RigidBodyDynamics


class HumanoidDynamics(RigidBodyDynamics):
    def __init__(self, urdf_file, config_file, mesh_dir=None):
        super(HumanoidDynamics, self).__init__(urdf_file, config_file, mesh_dir, floating_base=True)
        
    def _contact_block_dim(self) -> int:
        # Full 6D wrench per contact: [n; f] = [moment; force]
        return 6
    
    def _active_contact_frames(self, cnt_schedule: np.ndarray) -> list[int]:
        mask = cnt_schedule.astype(bool)
        if mask.size != self.nb_ee:
            raise ValueError(f"contact_schedule length {mask.size} != nb_ee {self.nb_ee}")
        return [fid for fid, active in zip(self.endeff_ids, mask) if active]

    def _compute_J_c(self, cnt_schedule: np.ndarray) -> np.ndarray:
        """
        Stacked contact Jacobian for active contacts.
        Shape: (6*m, nv). Uses the full 6xnv frame Jacobian J6, whose rows are [angular; linear].
        Requires update_fk(...) to have been called beforehand.
        """
        frames = self._active_contact_frames(cnt_schedule)
        m = len(frames)
        k = self._contact_block_dim()
        J_c = np.zeros((k * m, self.nv))
        for j, fid in enumerate(frames):
            J6 = pin.getFrameJacobian(self.rmodel, self.rdata, fid, pin.LOCAL_WORLD_ALIGNED)
            J_c[j*k:(j+1)*k, :] = J6
        return J_c
    
    def _compute_lambda_c(self, ee_wrenches: np.ndarray, cnt_schedule: np.ndarray) -> np.ndarray:
        """
        Stack the wrenches for active contacts.
        Expected ee_wrenches shape: (nb_ee*6,) or (nb_ee, 6), with ordering [n; f]
        Returns shape: (6*m,)
        """
        w = ee_wrenches.reshape(-1, 6)
        if w.shape[0] != self.nb_ee:
            raise ValueError(f"ee_wrenches has {w.shape[0]} rows, expected nb_ee={self.nb_ee}")
        mask = cnt_schedule.astype(bool)
        if mask.size != self.nb_ee:
            raise ValueError(f"contact_schedule length {mask.size} != nb_ee {self.nb_ee}")
        return w[mask].ravel()