import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MaxNLocator, FuncFormatter


# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'font.size' : 8,
#     'axes.unicode_minus' : False,
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

def read_data(path, motion_name, start, end):
    tau_meas = np.loadtxt(path+"/data/solo/paper/"+f"{motion_name}_tau_meas.dat", delimiter='\t', dtype=np.float32).T[start:end, :]
    tau_proj_llsq = np.loadtxt(path+"/data/solo/paper/"+f"{motion_name}_tau_proj_llsq.dat", delimiter='\t', dtype=np.float32)[start:end, :]
    tau_full_lmi = np.loadtxt(path+"/data/solo/paper/"+f"{motion_name}_tau_full_lmi.dat", delimiter='\t', dtype=np.float32)[start:end, :]
    tau_proj_lmi = np.loadtxt(path+"/data/solo/paper/"+f"{motion_name}_tau_proj_lmi.dat", delimiter='\t', dtype=np.float32)[start:end, :]
    tau_ped_nn = np.loadtxt(path+"/data/solo/paper/"+f"{motion_name}_tau_ped_nn.dat", delimiter='\t', dtype=np.float32)[start:end, :]
    return tau_meas, tau_proj_llsq, tau_full_lmi, tau_proj_lmi, tau_ped_nn

def plot(i, leg_idx, tau_meas, tau_proj_lmi, tau_ped_nn, motion_name):
    error_lmi = tau_meas - tau_proj_lmi
    error_nn = tau_meas - tau_ped_nn
    rmse_total_lmi = np.sqrt(np.mean(np.square(np.linalg.norm(error_lmi, axis=1))))
    rmse_total_nn = np.sqrt(np.mean(np.square(np.linalg.norm(error_nn, axis=1))))
    print(f'\n----{motion_name}---- Torque Prediction Errors ----')
    print(f'RMSE_LMI= {rmse_total_lmi}\nRMSE_NN={rmse_total_nn}')
    line_thick = 0.9
     # ----------------------- Hip Abduction/Adduction ----------------------- #
    axs[i,0].plot(t, tau_meas[:, leg_idx], "#90EE90", label="Meas", linewidth=1.6)
    axs[i,0].plot(t, tau_proj_lmi[:, leg_idx], "r--", label="LMI", linewidth=line_thick)
    axs[i,0].plot(t, tau_proj_llsq[:, leg_idx], "k--", label="SVD", linewidth=line_thick)
    axs[i,0].plot(t, tau_ped_nn[:, leg_idx], "b--", label="MLP", linewidth=line_thick)
    axs[i,0].get_yaxis().set_label_coords(-0.19,0.5)
    if i < 2:
        axs[i,0].xaxis.set_ticklabels([])
    if i == 0:
        axs[i,0].set(ylabel='Validataion \nTorque (Nm)')
        axs[i,0].get_xaxis().set_label_coords(0.5,0.9)
        axs[i,0].set_xlabel(leg+'-Hip Ab/Ad\n')
        # axs[i,0].legend(loc="upper right", shadow=True, fontsize="xx-small")
        axs[i,0].xaxis.set_label_position('top')
    if i == 1:
        axs[i,0].set(ylabel='New Trot\nTorque (Nm)')
    if i == 2:
        axs[i,0].set(ylabel='New Task (Jump)\nTorque (Nm)')
        axs[i,0].set_xlabel('Time (s)\n')
        axs[i,0].xaxis.set_label_position('bottom')
        axs[i,0].get_xaxis().set_label_coords(0.5,-0.17)

    # ----------------------- Hip Flexion/Extension ----------------------- #
    axs[i,1].plot(t, tau_meas[:, leg_idx+1], "#90EE90", label="Measured", linewidth=1.6)
    axs[i,1].plot(t, tau_proj_lmi[:, leg_idx+1], "r--", label="LMI (ours)", linewidth=line_thick)
    axs[i,1].plot(t, tau_proj_llsq[:, leg_idx+1], "k--", label="SVD", linewidth=line_thick)
    axs[i,1].plot(t, tau_ped_nn[:, leg_idx+1], "b--", label="MLP", linewidth=line_thick)
    if i < 2:
        axs[i,1].xaxis.set_ticklabels([])
    if i == 0:
        axs[i,1].legend(
            loc="upper right", 
            shadow=False, 
            fontsize="x-small", 
            bbox_to_anchor=(1.02, 0.45), 
            facecolor="white", 
            framealpha=1,
            # handlelength=1,      # Length of the legend handles (symbols)
            # handletextpad=0.5,   # Space between the legend handle and text
            # borderaxespad=0.1,   # Padding between the legend and the axes
            # borderpad=0.1,        # Padding inside the legend box
            labelspacing=0.1,    # Spacing between labels
            columnspacing=1.0 
        )
        axs[i,1].get_xaxis().set_label_coords(0.5,0.9)  
        axs[i,1].xaxis.set_label_position('top')
        axs[i,1].set_xlabel(leg+'-Hip Fl/Ex\n')
    if i == 2:
        axs[i,1].set_xlabel('Time (s)\n')
        axs[i,1].xaxis.set_label_position('bottom')
        axs[i,1].get_xaxis().set_label_coords(0.5,-0.17)

    # # ----------------------- Knee Flexion/Extension ----------------------- #
    axs[i,2].plot(t, tau_meas[:, leg_idx+2], "#90EE90", label="Meas", linewidth=1.6)
    axs[i,2].plot(t, tau_proj_lmi[:, leg_idx+2], "r--", linestyle="--", label="LMI (ours)", linewidth=line_thick)
    axs[i,2].plot(t, tau_proj_llsq[:, leg_idx+2], "k--", label="SVD", linewidth=line_thick)
    axs[i,2].plot(t, tau_ped_nn[:, leg_idx+2], "b--", label="NN", linewidth=line_thick)
    if i < 2:
        axs[i,2].xaxis.set_ticklabels([])
    if i == 0:
        axs[i,2].get_xaxis().set_label_coords(0.5,0.9)    
        axs[i,2].xaxis.set_label_position('top')
        axs[i,2].set_xlabel(leg+'-Knee Fl/Ex\n')
    if i == 2:
        axs[i,2].set_xlabel('Time (s)\n')
        axs[i,2].xaxis.set_label_position('bottom')
        axs[i,2].get_xaxis().set_label_coords(0.5,-0.17) 
    
    
if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.dirname(dir_path) # Root directory of the workspace
    
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            # axs[i,j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            # axs[i,j].yaxis.set_major_formatter(FuncFormatter(custom_y_formatter))  # Apply custom formatting to y-axis
            axs[i,j].xaxis.set_major_locator(MaxNLocator(integer=True))  # This line ensures x-axis ticks are integers
    
    # Initialize the time vector
    start = 100
    end = 200
    simulation_time = end - start
    t = np.arange(simulation_time) / 100
    
    # Indecis for motors in one leg
    leg_idx = 0
    leg = "FL"

    # Validation
    motion_name = "eval"
    tau_meas, tau_proj_llsq, tau_full_lmi, tau_proj_lmi, tau_ped_nn = read_data(path, motion_name, start, end)
    plot(0, 6, tau_meas, tau_proj_lmi, tau_ped_nn, motion_name)
    
    # Trot
    start = 200
    end = 300
    motion_name = "eval_trot"
    tau_meas, tau_proj_llsq, tau_full_lmi, tau_proj_lmi, tau_ped_nn = read_data(path, motion_name, start, end)
    plot(1, leg_idx, tau_meas, tau_proj_lmi, tau_ped_nn, motion_name)
    
    # Jump
    start = 200
    end = 300
    motion_name = "eval_jump"
    tau_meas, tau_proj_llsq, tau_full_lmi, tau_proj_lmi, tau_ped_nn = read_data(path, motion_name, start, end)
    plot(2, 0, tau_meas, tau_proj_lmi, tau_ped_nn, motion_name)
    
    # Show the plot
    fig.set_size_inches(w=7.05, h=2.95)
    fig.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.09, bottom=0.08, right=0.99, top=0.95, wspace=0.19, hspace=0.1)
    plt.show()

    # Save as pgf files
    # plt.savefig("/home/khorshidi/Documents/3_Papers-Conferences/2024_09_ICRA/files/"+"solo_fl_torque.pgf")