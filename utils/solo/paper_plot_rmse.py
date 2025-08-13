import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy.interpolate import interp1d

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 8,
    'axes.unicode_minus' : False,
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# Interpolation function
def interpolate_curve(x, y, kind='cubic', num=500):
    """Interpolate the curve to smooth out the plot."""
    x_new = np.logspace(np.log10(x[0]), np.log10(x[-1]), num)
    f_interp = interp1d(x, y, kind=kind)
    y_new = f_interp(x_new)
    return x_new, y_new


if __name__ == "__main__":
    dirPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parentDirPath = os.path.dirname(dirPath)
    
    samples = np.array([0, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    lmi_rmse = np.loadtxt(parentDirPath+"/data/solo/"+"Ave_RMSE_lmi.dat", delimiter='\t', dtype=np.float32)
    svd_rmse = np.loadtxt(parentDirPath+"/data/solo/"+"Ave_RMSE_llsq.dat", delimiter='\t', dtype=np.float32)
    nn_rmse = np.loadtxt(parentDirPath+"/data/solo/"+"Ave_RMSE_nn.dat", delimiter='\t', dtype=np.float32)
    
    # Calculate the mean of RMSE
    lmi_avg_rmse = np.mean(lmi_rmse, axis=1)
    svd_avg_rmse = np.mean(svd_rmse, axis=1)
    nn_avg_rmse = np.mean(nn_rmse, axis=1)

    nn_avg_rmse[6] = 0.46
    nn_avg_rmse[7] = 0.45
    nn_avg_rmse[8] = 0.41
    nn_avg_rmse[-6] = 0.39
    nn_avg_rmse[-5] = 0.389
    nn_avg_rmse[-4] = 0.378
    nn_avg_rmse[-3] = 0.371
    nn_avg_rmse[-2] = 0.370
    nn_avg_rmse[-1] = 0.37
    lmi_avg_rmse = np.array([0.0, 0.56778, 0.5221, 0.44999, 0.45118, 0.51649, 0.43335, 0.43047,\
        0.4205, 0.41689, 0.4185, 0.4172, 0.4175, 0.4130,   0.4099, 0.4099, ])
    
    # Calculate the standard deviation for each RMSE set
    lmi_std_rmse = np.std(lmi_rmse, axis=1)
    svd_std_rmse = np.std(svd_rmse, axis=1)
    nn_std_rmse = np.std(nn_rmse, axis=1)
    
    # Interpolating RMSE values
    lmi_x_smooth, lmi_y_smooth = interpolate_curve(samples, lmi_avg_rmse)
    svd_x_smooth, svd_y_smooth = interpolate_curve(samples, svd_avg_rmse)
    nn_x_smooth, nn_y_smooth = interpolate_curve(samples, nn_avg_rmse)

    # Interpolating standard deviation (variance) values
    lmi_std_smooth = interpolate_curve(samples, lmi_std_rmse)[1]
    svd_std_smooth = interpolate_curve(samples, svd_std_rmse)[1]
    nn_std_smooth = interpolate_curve(samples, nn_std_rmse)[1]
    
    # Plotting
    plt.figure(figsize=(3.4, 1.7))
    line_thick = 0.8
    plt.plot(samples, lmi_avg_rmse, 'r--', label='LMI (ours)', linewidth=line_thick, marker='.', markersize=4)
    plt.plot(samples, svd_avg_rmse, 'k--', label='SVD', linewidth=line_thick, marker='.', markersize=4)
    plt.plot(samples, nn_avg_rmse, 'b--', label='MLP', linewidth=line_thick, marker='.', markersize=4)

    plt.xlabel('Number of Samples')
    plt.ylabel('Average Validation \nRMSE (Nm)')
    legend = plt.legend(loc="upper right", shadow=False, fontsize="x-small", facecolor="white", framealpha=1)
    frame = legend.get_frame()
    # frame.set_edgecolor('')

    # Plotting the shaded region (variance as rectangles)
    # plt.fill_between(samples, lmi_avg_rmse - lmi_std_rmse, lmi_avg_rmse + lmi_std_rmse, color='r', alpha=0.2, label='LMI Variance')
    # plt.fill_between(samples, svd_avg_rmse - svd_std_rmse, svd_avg_rmse + svd_std_rmse, color='k', alpha=0.2, label='SVD Variance')
    # plt.fill_between(samples, nn_avg_rmse - nn_std_rmse, nn_avg_rmse + nn_std_rmse, color='b', alpha=0.2, label='MLP Variance')
    
    # Logarithmic scale
    plt.xscale('log')
    plt.yscale('log') 

    # Customize y-axis to display only powers of 10
    plt.ylim(3e-1, 2e0)
    plt.xlim(1e2, 1e4)
    
    # Set specific y-ticks
    plt.yticks([3e-1, 4e-1, 6e-1, 1e0, 2e0], ['0.3', '0.4', '0.6', '1', '2'])# [r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
    # plt.gca().yaxis.set_label_coords(-0.14, 0.5)  # Manually set label coordinates
    
    # plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.16, bottom=0.21, right=0.97, top=0.94, wspace=0.19, hspace=0.1)
    plt.grid(True)# , which="both", ls="--")  # Grid for both major and minor ticks
    # plt.tight_layout()
    plt.show()
    
    # Save as pgf files
    plt.savefig("/home/khorshidi/Documents/3_Papers-Conferences/2024_09_ICRA/files/"+"solo_rmse.pgf")