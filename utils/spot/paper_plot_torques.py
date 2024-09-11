import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MaxNLocator, FuncFormatter


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size' : 8,
    'axes.unicode_minus' : False,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def read_data(path, motion_name, start, end):
    tau_meas = np.loadtxt(path+"/data/spot/paper/"+f"{motion_name}_tau_meas.dat", delimiter='\t', dtype=np.float32)[start:end, :]
    tau_proj_lmi = np.loadtxt(path+"/data/spot/paper/"+f"{motion_name}_tau_proj_lmi.dat", delimiter='\t', dtype=np.float32)[start:end, :]
    return tau_meas, tau_proj_lmi

def plot(i, leg_idx, tau_meas, tau_proj_lmi, motion_name):
    error_lmi = tau_meas - tau_proj_lmi
    rmse_total_lmi = np.sqrt(np.mean(np.square(np.linalg.norm(error_lmi, axis=1))))
    print(f'\n----{motion_name}---- Torque Prediction Errors ----')
    print(f'RMSE_LMI= {rmse_total_lmi}\nRMSE_NN={rmse_total_lmi}')
    line_thick = 0.9
     # ----------------------- Hip Abduction/Adduction ----------------------- #
    axs[i,0].plot(t, tau_meas[:, leg_idx], "#90EE90", label="Meas", linewidth=1.3)
    axs[i,0].plot(t, tau_proj_lmi[:, leg_idx], "r-.", label="LMI", linewidth=line_thick)
    axs[i,0].get_yaxis().set_label_coords(-0.15,0.5)
    if i < 1:
        axs[i,0].xaxis.set_ticklabels([])
    if i == 0:
        axs[i,0].set(ylabel='Validataion (Crawl)\nTorque (Nm)')
        axs[i,0].get_xaxis().set_label_coords(0.5,0.95)
        axs[i,0].set_xlabel(leg+'-Hip Ab/Ad\n')
        axs[i,0].legend(loc="lower right", shadow=True, fontsize="x-small")
        axs[i,0].xaxis.set_label_position('top')
    if i == 1:
        axs[i,0].set(ylabel='New Task (Walk)\nTorque (Nm)')
        axs[i,0].set_xlabel('Time (s)\n')
        axs[i,0].xaxis.set_label_position('bottom')
        axs[i,0].get_xaxis().set_label_coords(0.5,-0.22)

    # ----------------------- Hip Flexion/Extension ----------------------- #
    axs[i,1].plot(t, tau_meas[:, leg_idx+1], "#90EE90", label="Meas", linewidth=1.3)
    axs[i,1].plot(t, tau_proj_lmi[:, leg_idx+1], "r-.", label="LMI", linewidth=line_thick)
    if i < 1:
        axs[i,1].xaxis.set_ticklabels([])
    if i == 0:
        axs[i,1].get_xaxis().set_label_coords(0.5,0.95)  
        axs[i,1].xaxis.set_label_position('top')
        axs[i,1].set_xlabel(leg+'-Hip Fl/Ex\n')
    if i == 1:
        axs[i,1].set_xlabel('Time (s)\n')
        axs[i,1].xaxis.set_label_position('bottom')
        axs[i,1].get_xaxis().set_label_coords(0.5,-0.22)

    # # ----------------------- Knee Flexion/Extension ----------------------- #
    axs[i,2].plot(t, tau_meas[:, leg_idx+2], "#90EE90", label="Meas", linewidth=1.3)
    axs[i,2].plot(t, tau_proj_lmi[:, leg_idx+2], "r-.", label="LMI", linewidth=line_thick)
    if i < 1:
        axs[i,2].xaxis.set_ticklabels([])
    if i == 0:
        axs[i,2].get_xaxis().set_label_coords(0.5,0.95)    
        axs[i,2].xaxis.set_label_position('top')
        axs[i,2].set_xlabel(leg+'-Knee Fl/Ex\n')
    if i == 1:
        axs[i,2].set_xlabel('Time (s)\n')
        axs[i,2].xaxis.set_label_position('bottom')
        axs[i,2].get_xaxis().set_label_coords(0.5,-0.22)
        
if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.dirname(dir_path) # Root directory of the workspace
    
    fig, axs = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            # axs[i,j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            # axs[i,j].yaxis.set_major_formatter(FuncFormatter(custom_y_formatter))  # Apply custom formatting to y-axis
            axs[i,j].xaxis.set_major_locator(MaxNLocator(integer=True))  # This line ensures x-axis ticks are integers
    
    # Initialize the time vector
    simulation_time = 500
    t = np.arange(simulation_time) / 100
    
    # Indecis for motors in one leg
    leg_idx = 0
    leg = "FL"
    
    # Crawl validate
    start = 1500
    end = 2000
    motion_name = "spot_crawl"
    plot_idx = 0
    tau_meas, tau_proj_lmi = read_data(path, motion_name, start, end)
    plot(plot_idx, leg_idx, tau_meas, tau_proj_lmi, motion_name)

    # Walk
    start = 2300
    end = 2800
    motion_name = "spot_walk"
    plot_idx = 1
    tau_meas, tau_proj_lmi = read_data(path, motion_name, start, end)
    plot(plot_idx, leg_idx, tau_meas, tau_proj_lmi, motion_name)
    
    # Show the plot
    fig.set_size_inches(w=7.05, h=2.7)
    fig.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.08, bottom=0.13, right=0.99, top=0.92, wspace=0.19, hspace=0.1)
    plt.show()
    
    # Save as pgf files
    plt.savefig("/home/khorshidi/Documents/3_Papers-Conferences/2024_09_ICRA/files/"+"spot_fl_torque.pgf")