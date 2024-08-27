import numpy as np
import pinocchio as pin

r = 0.2
p = 0.1
y = -0.15

R_x = np.array([[1, 0        , 0         ],
                [0, np.cos(r), -np.sin(r)],
                [0, np.sin(r), np.cos(r)]]
               )
R_y = np.array([[np.cos(p) , 0, np.sin(p)],
                [0         , 1, 0        ],
                [-np.sin(p), 0, np.cos(p)]])

R_z = np.array([[np.cos(y), -np.sin(y), 0],
                [np.sin(y), np.cos(y) , 0],
                [0        , 0         , 1]])
# Extrinsic rotation (rotation around fixed axes)
# Order: first:roll, then:pith, and:yaw 
R_1 = R_z@R_y@R_x
R_2 = pin.utils.rpyToMatrix(r, p, y)
# print(R_1 - R_2)


com = np.array([1, -2, 3])
print(pin.skew(com) @ pin.skew(com).T)
print(pin.skew(-com) @ pin.skew(-com).T)