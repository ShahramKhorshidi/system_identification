import numpy as np
import matplotlib.pyplot as plt

class PlotClass():
    def __init__(self, phi_prior):
        self._phi_prior = phi_prior
    
    def plot_mass(self, phi_ident, title):
        # Extract the mass values (every 10th element starting from the 0th element)
        masses_prior = self._phi_prior[::10]
        masses_calculated = phi_ident[::10]

        # Create an array representing the link indices
        link_indices = np.arange(13)  # Links are numbered 1 to 13

        # Plot the bar diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.get_current_fig_manager().set_window_title(title)
        
        bar_width = 0.35  # Width of the bars
        index = np.arange(len(link_indices))

        bar1 = ax.bar(index, masses_prior, bar_width, label='Actual Mass')
        bar2 = ax.bar(index + bar_width, masses_calculated, bar_width, label='Identified Mass')

        # Adding labels, title and legend
        ax.set_xlabel('Link Number')
        ax.set_ylabel('Mass (kg)')
        ax.set_title('Masses for all links')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(link_indices)
        ax.legend()
        plt.tight_layout()

    def plot_h(self, phi_ident, title):
        num_links = 13
        num_params_per_link = 10
        
        # Initialize arrays to store h_x, h_y, h_z values
        h_x_actual = []
        h_y_actual = []
        h_z_actual = []
        h_x_predicted = []
        h_y_predicted = []
        h_z_predicted = []

        # Extract actual and predicted values
        for i in range(num_links):
            idx = i * num_params_per_link
            h_x_actual.append(self._phi_prior[idx + 1])
            h_y_actual.append(self._phi_prior[idx + 2])
            h_z_actual.append(self._phi_prior[idx + 3])
            
            h_x_predicted.append(phi_ident[idx + 1])
            h_y_predicted.append(phi_ident[idx + 2])
            h_z_predicted.append(phi_ident[idx + 3])

        # Plotting
        index = np.arange(num_links)
        bar_width = 0.13

        fig, axs = plt.subplots(3, 1, figsize=(12, 18))
        plt.get_current_fig_manager().set_window_title(title)
        
        # Plot h_x
        axs[0].bar(index - bar_width/2, h_x_actual, bar_width, label='Actual h_x')
        axs[0].bar(index + bar_width/2, h_x_predicted, bar_width, label='Identified h_x')
        axs[0].set_xticks(index)
        axs[0].set_xticklabels(range(num_links))
        axs[0].set_ylabel('h_x (kg.m)')
        axs[0].legend()

        # Plot x_y
        axs[1].bar(index - bar_width/2, h_y_actual, bar_width, label='Actual h_y')
        axs[1].bar(index + bar_width/2, h_y_predicted, bar_width, label='Identified h_h')
        axs[1].set_xticks(index)
        axs[1].set_xticklabels(range(num_links))
        axs[1].set_ylabel('h_y (kg.m)')
        axs[1].legend()

        # Plot h_z
        axs[2].bar(index - bar_width/2, h_z_actual, bar_width, label='Actual h_z')
        axs[2].bar(index + bar_width/2, h_z_predicted, bar_width, label='Identified h_z')
        axs[2].set_xticks(index)
        axs[2].set_xticklabels(range(num_links))
        axs[2].legend()
        axs[2].set_ylabel('h_z (kg.m)')
        axs[2].set_xlabel('Link Number')
        plt.tight_layout()

    def plot_inertia(self, phi_ident, title):
        num_links = 13
        num_params_per_link = 10
        
        # Initialize arrays to store h_x, h_y, h_z values
        I_xx_actual = []
        I_xy_actual = []
        I_xz_actual = []
        I_yy_actual = []
        I_yz_actual = []
        I_zz_actual = []
        I_xx_predicted = []
        I_xy_predicted = []
        I_xz_predicted = []
        I_yy_predicted = []
        I_yz_predicted = []
        I_zz_predicted = []

        # Extract actual and predicted values
        for i in range(num_links):
            idx = i * num_params_per_link
            I_xx_actual.append(self._phi_prior[idx + 4])
            I_xy_actual.append(self._phi_prior[idx + 5])
            I_yy_actual.append(self._phi_prior[idx + 6])
            I_xz_actual.append(self._phi_prior[idx + 7])
            I_yz_actual.append(self._phi_prior[idx + 8])
            I_zz_actual.append(self._phi_prior[idx + 9])
                    
            I_xx_predicted.append(phi_ident[idx + 4])
            I_xy_predicted.append(phi_ident[idx + 5])
            I_yy_predicted.append(phi_ident[idx + 6])
            I_xz_predicted.append(phi_ident[idx + 7])
            I_yz_predicted.append(phi_ident[idx + 8])
            I_zz_predicted.append(phi_ident[idx + 9])
            
        # Plotting
        index = np.arange(num_links)
        bar_width = 0.13

        fig, axs = plt.subplots(3, 1, figsize=(12, 18))
        plt.get_current_fig_manager().set_window_title(title)
        
        # Plot I_xx
        axs[0].bar(index - bar_width/2, I_xx_actual, bar_width, label='Actual I_xx')
        axs[0].bar(index + bar_width/2, I_xx_predicted, bar_width, label='Identified I_xx')
        # axs[0].set_xlabel('Link')
        # axs[0].set_ylabel('Values')
        # axs[0].set_title('I_xx for all links')
        axs[0].set_xticks(index)
        axs[0].set_xticklabels(range(num_links))
        axs[0].set_ylabel('I_xx (kg.m^2)')
        axs[0].legend()

        # Plot I_xy
        axs[1].bar(index - bar_width/2, I_xy_actual, bar_width, label='Actual I_xy')
        axs[1].bar(index + bar_width/2, I_xy_predicted, bar_width, label='Identified I_xy')
        axs[1].set_xticks(index)
        axs[1].set_xticklabels(range(num_links))
        axs[1].set_ylabel('I_xy (kg.m^2)')
        axs[1].legend()

        # Plot I_xz
        axs[2].bar(index - bar_width/2, I_xz_actual, bar_width, label='Actual I_xz')
        axs[2].bar(index + bar_width/2, I_xz_predicted, bar_width, label='Identified I_xz')
        axs[2].set_xticks(index)
        axs[2].set_xticklabels(range(num_links))
        axs[2].set_xlabel('Link Number')
        axs[2].set_ylabel('I_xz (kg.m^2)')
        axs[2].legend()
        
        fig2, axs2 = plt.subplots(3, 1, figsize=(12, 18))
        plt.get_current_fig_manager().set_window_title(title)
        
        # Plot I_yy
        axs2[0].bar(index - bar_width/2, I_yy_actual, bar_width, label='Actual I_yy')
        axs2[0].bar(index + bar_width/2, I_yy_predicted, bar_width, label='Identified I_yy')
        axs2[0].set_xticks(index)
        axs2[0].set_xticklabels(range(num_links))
        axs2[0].set_ylabel('I_yy (kg.m^2)')
        axs2[0].legend()

        # Plot I_yz
        axs2[1].bar(index - bar_width/2, I_yz_actual, bar_width, label='Actual I_yz')
        axs2[1].bar(index + bar_width/2, I_yz_predicted, bar_width, label='Identified I_yz')
        axs2[1].set_xticks(index)
        axs2[1].set_xticklabels(range(num_links))
        axs2[1].set_ylabel('I_yz (kg.m^2)')
        axs2[1].legend()

        # Plot I_zz
        axs2[2].bar(index - bar_width/2, I_zz_actual, bar_width, label='Actual I_zz')
        axs2[2].bar(index + bar_width/2, I_zz_predicted, bar_width, label='Identified I_zz')
        axs2[2].set_xticks(index)
        axs2[2].set_xticklabels(range(num_links))
        axs2[2].set_xlabel('Link Number')
        axs2[2].set_ylabel('I_zz (kg.m^2)')
        axs2[2].legend()
        plt.tight_layout()

    def proj_torques_null(self, q, dq, ddq, cnt, torque, b_v, b_c, phi, sys_idnt):
        tau_nn_proj = []
        # For each data ponit we calculate the rgeressor and torque vector, and stack them
        for i in range(q.shape[1]):
            sys_idnt.update_fk(q[:, i], dq[:, i], ddq[:, i])
            Y = sys_idnt.get_regressor_matrix(q[:, i], dq[:, i], ddq[:, i])
            P = sys_idnt.get_null_space_proj(cnt[:, i])
            tau_meas = P @ sys_idnt.S.T @ torque[:, i]
            tau_nn_proj.append(tau_meas[6:])
        
        tau_nn_proj = np.vstack(tau_nn_proj)
        return tau_nn_proj
    
    def plot_proj_torques(self, q, dq, ddq, cnt, torque, b_v, b_c, phi, sys_idnt, title):
        predicted = []
        measured = []
        # For each data ponit we calculate the rgeressor and torque vector, and stack them
        for i in range(q.shape[1]):
            sys_idnt.update_fk(q[:, i], dq[:, i], ddq[:, i])
            Y = sys_idnt.get_regressor_matrix(q[:, i], dq[:, i], ddq[:, i])
            P = sys_idnt.get_null_space_proj(cnt[:, i])
            tau_pred = P @ (Y @ phi 
                            - sys_idnt.S.T @ (np.diag(b_v) @ dq[sys_idnt.base_dof:, i] 
                            + np.diag(b_c) @ np.sign(dq[sys_idnt.base_dof:, i])))
            tau_meas = P @ sys_idnt.S.T @ torque[:, i]
            predicted.append(tau_pred[6:])
            measured.append(tau_meas[6:])

        predicted = np.vstack(predicted)
        measured = np.vstack(measured)
        
        error = measured - predicted
        rmse_total = np.sqrt(np.mean(np.square(np.linalg.norm(error, axis=1)))) # overall RMSE
        joint_tau_rmse = np.sqrt(np.mean(np.square(error), axis=0)) # RMSE for each joint
        print(f'\n-------------------- {title} parameters --------------------')
        print(f'Torque Prediction Errors: RMSE_total= {rmse_total}\nRMSE_per_joints={joint_tau_rmse}')
        
        num_joints = measured.shape[1]
        rows = 4
        cols = 3
        
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 15))
        plt.get_current_fig_manager().set_window_title(title)
        
        for j in range(num_joints):
            ax = axes[j // cols, j % cols]
            
            ax.plot(measured[:, j], label='Meaured', color='green', linestyle='--')
            ax.plot(predicted[:, j], label='Identified', color='red', linestyle='--')
            
            if j == 0:
                ax.set_title("Hip Abduction/Adduction")
            if j == 1:
                ax.set_title("Hip Flexion/Extension")
            if j == 2:
                ax.set_title("Knee Flexion/Extension")
            ax.set_ylabel('Torque (Nm)')
            if j>=9:
                ax.set_xlabel('Sample')
            ax.legend()
        axes[0][0].set_ylabel('FL - Torque (Nm)')
        axes[1][0].set_ylabel('FR - Torque (Nm)')
        axes[2][0].set_ylabel('HL - Torque (Nm)')
        axes[3][0].set_ylabel('HR - Torque (Nm)')
        plt.tight_layout()
        return measured, predicted
    
    def plot_solo_torques(self, q, dq, ddq, cnt, torque, b_v, b_c, phi, sys_idnt, title, force):
        predicted = []
        measured = []
        # For each data ponit we calculate the rgeressor and torque vector, and stack them
        for i in range(q.shape[1]):
            sys_idnt.update_fk(q[:, i], dq[:, i], ddq[:, i])
            Y = sys_idnt.get_regressor_matrix(q[:, i], dq[:, i], ddq[:, i])
            tau_pred = (Y @ phi 
                        - sys_idnt.S.T @ (np.diag(b_v) @ dq[sys_idnt.base_dof:, i] 
                        + np.diag(b_c) @ np.sign(dq[sys_idnt.base_dof:, i])))
            F = sys_idnt.get_cnt_force(force[:, i], cnt[:, i])
            tau_pred -= F
            predicted.append(tau_pred[6:])
            measured.append(torque[:, i])
        
        predicted = np.vstack(predicted)
        measured = np.vstack(measured)
        
        error = measured - predicted
        rmse_total = np.sqrt(np.mean(np.square(np.linalg.norm(error, axis=1)))) # overall RMSE
        joint_tau_rmse = np.sqrt(np.mean(np.square(error), axis=0)) # RMSE for each joint
        print(f'\n-------------------- {title} parameters --------------------')
        print(f'Torque Prediction Errors: RMSE_total= {rmse_total}\nRMSE_per_joints={joint_tau_rmse}')
        
        num_joints = measured.shape[1]
        rows = 4
        cols = 3
        
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 15))
        plt.get_current_fig_manager().set_window_title(title)
        
        for j in range(num_joints):
            ax = axes[j // cols, j % cols]
            
            ax.plot(measured[:, j], label='Meaured', color='green', linestyle='--')
            ax.plot(predicted[:, j], label='Identified', color='red', linestyle='--')
            
            if j == 0:
                ax.set_title("Hip Abduction/Adduction")
            if j == 1:
                ax.set_title("Hip Flexion/Extension")
            if j == 2:
                ax.set_title("Knee Flexion/Extension")
            ax.set_ylabel('Torque (Nm)')
            if j>=9:
                ax.set_xlabel('Sample')
            ax.legend()
        axes[0][0].set_ylabel('FL - Torque (Nm)')
        axes[1][0].set_ylabel('FR - Torque (Nm)')
        axes[2][0].set_ylabel('HL - Torque (Nm)')
        axes[3][0].set_ylabel('HR - Torque (Nm)')
        plt.tight_layout()
        return predicted

    def plot_nn_torques(self, tau, tau_nn, title):
        error = tau - tau_nn
        rmse_total = np.sqrt(np.mean(np.square(np.linalg.norm(error, axis=1)))) # overall RMSE
        joint_tau_rmse = np.sqrt(np.mean(np.square(error), axis=0)) # RMSE for each joint
        print(f'\n-------------------- {title} parameters --------------------')
        print(f'Torque Prediction Errors: RMSE_total= {rmse_total}\nRMSE_per_joints={joint_tau_rmse}')
        num_joints = tau.shape[1]
        rows = 4
        cols = 3
        
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 15))
        plt.get_current_fig_manager().set_window_title(title)
        
        for j in range(num_joints):
            ax = axes[j // cols, j % cols]
            
            ax.plot(tau[:, j], label='Meaured', color='green', linestyle='--')
            ax.plot(tau_nn[:, j], label='Identified', color='red', linestyle='--')
            
            if j == 0:
                ax.set_title("Hip Abduction/Adduction")
            if j == 1:
                ax.set_title("Hip Flexion/Extension")
            if j == 2:
                ax.set_title("Knee Flexion/Extension")
            ax.set_ylabel('Torque (Nm)')
            if j>=9:
                ax.set_xlabel('Sample')
            ax.legend()
        axes[0][0].set_ylabel('FL - Torque (Nm)')
        axes[1][0].set_ylabel('FR - Torque (Nm)')
        axes[2][0].set_ylabel('HL - Torque (Nm)')
        axes[3][0].set_ylabel('HR - Torque (Nm)')
        plt.tight_layout()
        
    def plot_eigval(self, I_bar, I, J, C, trace, title):
        num_links = 13

        # Plot the bar diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.get_current_fig_manager().set_window_title(title)
        
        bar_width = 0.15  # Width of the bars
        index = np.arange(num_links)

        bar1 = ax.bar(index - bar_width, I_bar, bar_width, label='Inertia matrix')
        bar2 = ax.bar(index, I, bar_width, label='Spatial body inerta matrix')
        bar3 = ax.bar(index + bar_width, J, bar_width, label='Pseudo inertia matrix')
        bar4 = ax.bar(index + 2*bar_width, C, bar_width, label='CoM constraint matrix')
        # bar5 = ax.bar(index + 3*bar_width, trace, bar_width, label='trace')
        
        # Adding labels, title and legend
        ax.set_xlabel('Link Number')
        ax.set_ylabel('Minimum Eigen Value')
        ax.set_title('Minimum Eigenvalue for different matrices')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(index)
        ax.legend()
        plt.tight_layout()