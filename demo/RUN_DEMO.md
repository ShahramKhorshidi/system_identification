### How to run the demo
1. Create a data folder in your repository.
2. Copy the provided files into this folder.
This data includes 24,000 samples of the Spot robot performing various trajectories, such as base wobbling, squatting with all feet in contact, forward-backward walking, and side-to-side walking.
```
python spot_identification.py
```
After solving the optimization, the output will look similar to the following, showing the identified parameters for each link:
```
--------------- Inertial Parameters of "front_left_hip" ---------------
|Parameter    |A priori     |Identified   |Change       |error %      |
|mass (kg)    |     1.680000|     1.789420|     0.109420|          6.5|
|c_x (m)      |    -0.005374|    -0.005521|    -0.000147|         -2.7|
|c_y (m)      |     0.012842|     0.013002|     0.000160|          1.2|
|c_z (m)      |     0.000099|     0.000158|     0.000059|         60.0|
|I_xx (kg.m^2)|     0.002391|     0.002413|     0.000022|          0.9|
|I_xy (kg.m^2)|     0.000126|     0.000139|     0.000012|          9.7|
|I_xz (kg.m^2)|    -0.000009|    -0.000008|     0.000001|          6.2|
|I_yy (kg.m^2)|     0.002057|     0.002061|     0.000004|          0.2|
|I_yz (kg.m^2)|     0.000222|     0.000220|    -0.000002|         -1.0|
|I_zz (kg.m^2)|     0.002396|     0.002425|     0.000029|          1.2|
```