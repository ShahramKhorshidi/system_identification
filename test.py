import numpy as np 
import pinocchio as pin
a = np.array([1, 2 , 3])
crf = pin.crf(a)
print(crf)
