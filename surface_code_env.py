'''
This module contains the implementation of the surface code's environment
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Patch

class SurfaceCode:
    def __init__(self, d, p_phys, p_meas=0, error_model='X', volume_depth=1):

        if d % 2 == 0:
            raise ValueError("The code distance for the rotated surface code must be odd.")
        
        self.d = d
        self.p_phys = p_phys
        self.p_meas = p_meas
        self.error_model = error_model
        self.volume_depth = volume_depth
        self._initialize_environment()


    def _initialize_environment(self):

        self.data_qubits_coord, self.x_stabs_coord, self.z_stabs_coord = self._assign_qubit_coordinates()
        self.data_mask, self.x_mask, self.z_mask = self._create_masks()
        self.hidden_state, self.syndrome_lattice = self._simulate_errors()
        self.action_history = np.zeros((2*self.d+1, 2*self.d+1, 2))
        self.visible_state = np.stack([self.x_mask, self.z_mask, self.syndrome_lattice[:,:,0], 
                                       self.syndrome_lattice[:,:,1],self.data_mask, 
                                       self.action_history[:,:,0], self.action_history[:,:,1]])


    def _assign_qubit_coordinates(self):

        data_qubits_coord = [(i,j) for i in range(1, 2*self.d, 2) for j in range(1, 2*self.d, 2)]

        x_stabs_coord = []
        z_stabs_coord = []

        # Assign coordinates to stabilizers
        for i in range(0, 2*self.d+1, 2):
            for j in range(0, 2*self.d+1, 2):
                if (i % (2*self.d) == 0) or (j % (2*self.d) == 0):
                    # boundary logic
                    z_stab_left_cond = (j==0) and (i%4==0) and (i!=0)
                    z_stab_right_cond = (j==2*self.d) and ((i+2)%4==0) and (i!=2*self.d)
                    if z_stab_left_cond or z_stab_right_cond:
                        z_stabs_coord.append((i,j))

                    x_stab_top_cond = (i==0) and ((j+2)%4==0) and (j!=2*self.d)
                    x_stab_bottom_cond = (i==2*self.d) and (j%4==0) and (j!=0)
                    if x_stab_top_cond or x_stab_bottom_cond:
                        x_stabs_coord.append((i,j))

                else:
                    if (i/2+j/2) % 2 == 0:
                        z_stabs_coord.append((i,j))
                    else:
                        x_stabs_coord.append((i,j))

        return np.array(data_qubits_coord), np.array(x_stabs_coord), np.array(z_stabs_coord)


    def _create_masks(self):

        data_mask = np.zeros((2*self.d+1, 2*self.d+1))
        x_mask = np.zeros((2*self.d+1, 2*self.d+1))
        z_mask = np.zeros((2*self.d+1, 2*self.d+1))

        data_mask[1:2*self.d:2, 1:2*self.d:2] = 1
        x_mask[self.x_stabs_coord[:, 0], self.x_stabs_coord[:, 1]] = 1
        z_mask[self.z_stabs_coord[:, 0], self.z_stabs_coord[:, 1]] = 1
                
        return data_mask, x_mask, z_mask
    

    def _simulate_errors(self):

        # Initialize syndrome lattice
        syndrome_lattice_x = np.zeros((2*self.d+1, 2*self.d+1))
        syndrome_lattice_z = np.zeros((2*self.d+1, 2*self.d+1))

        if self.error_model == 'X':
            hidden_state_x = np.random.choice([-1,1], size=(self.d, self.d), p=[self.p_phys,1-self.p_phys])
            hidden_state_z = np.ones((self.d, self.d))
            
            # In the bit flip model, X stabilizers are not triggered 
            syndrome_lattice_x[self.x_stabs_coord[:, 0], self.x_stabs_coord[:, 1]] = 1

            # Trigger Z stabilizers if an odd number of support data qubits are bit-flipped
            for i, j in self.z_stabs_coord:
                support = self._obtain_support_qubits(i, j)
                syndrome_lattice_z[i,j] = np.prod(hidden_state_x[support[:,0], support[:,1]])

        elif self.error_model == 'depolarizing':
            
            # Randomly choose I,X,Y,Z according to probabilities
            choices = np.random.choice(
                [0, 1, 2, 3],      # 0=I, 1=X, 2=Z, 3=Y
                size=(self.d, self.d),
                p=[1 - self.p_phys, self.p_phys/3, self.p_phys/3, self.p_phys/3]
            )

            # For the X (Z) channel, fill in a -1 if there is an X (Z) or Y error 
            hidden_state_x = np.ones((self.d, self.d))
            hidden_state_z = np.ones((self.d, self.d))
            hidden_state_x[(choices == 1) | (choices == 3)] = -1   # X or Y
            hidden_state_z[(choices == 2) | (choices == 3)] = -1   # Z or Y

            # Trigger X stabilizers if an odd number of support data qubits are phase-flipped
            for i, j in self.x_stabs_coord:
                support = self._obtain_support_qubits(i, j)
                syndrome_lattice_x[i,j] = np.prod(hidden_state_x[support[:,0], support[:,1]])

            # Trigger X stabilizers if an odd number of support data qubits are bit-flipped
            for i, j in self.x_stabs_coord:
                support = self._obtain_support_qubits(i, j)
                syndrome_lattice_z[i,j] = np.prod(hidden_state_z[support[:,0], support[:,1]])
        
        # Stack the hidden states and syndrome lattices into single tensors
        hidden_state = np.stack([hidden_state_x, hidden_state_z], axis=-1)
        syndrome_lattice = np.stack([syndrome_lattice_x, syndrome_lattice_z], axis=-1)

        return hidden_state, syndrome_lattice

    def _obtain_support_qubits(self, i, j):

        if i == 0:
            support = np.array([[i+1,j+1],[i+1,j-1]])
        elif i == 2*self.d:
            support = np.array([[i-1,j+1], [i-1,j-1]])
        elif j == 0:
            support = np.array([[i+1,j+1], [i-1,j+1]])
        elif j == 2*self.d:
            support = np.array([[i+1,j-1], [i-1,j-1]])
        else: 
            support = np.array([[i+1,j+1],[i+1,j-1],[i-1,j+1],[i-1,j-1]])

        # Express support in data qubits coordinates (self.d, self.d)
        support = (support - 1) // 2

        return support
    

    def reset(self):
        
        self.hidden_state, self.syndrome_lattice = self._simulate_errors()
        self.hidden_state, self.syndrome_lattice = self._simulate_errors()
        self.action_history = np.zeros((2*self.d+1, 2*self.d+1, 2))
        self.visible_state = np.stack([self.x_mask, self.z_mask, self.syndrome_lattice,self.data_mask, self.action_history])
        
        return self.visible_state
        
    def step(self, action):
        '''
        Assume that action = [i,j,*] where * can be 0 (identity), 1 (X) or 2 (Z)
        '''
        done = False
        self.next_visible_state = np.copy(self.visible_state)
        if action[2] == 1:
            if self.visible_state[action[0],action[1],5] == 0:
                self.next_visible_state[action[0],action[1],5] = 1
                # Update hidden_state
                

            else:
                done = True
        elif action[2] == 1:
            if self.visible_state[action[0],action[1],6] == 0:
                self.next_visible_state[action[0],action[1],6] = 1
                # Update hidden state
            else:
                done = True

        # REWARD SYSTEM: 
        reward = 0

        # 1. Discount every step to make the decoder efficient
        reward -= 0.1


        # 2. Discount and finish episode if action is repeated
        if self.action_history[action[0], action[1], action[2]] == 1:
            reward -= 1

        # 3. Big reward if all syndromes are +1 and no logical error
        
        # 4. Big discount if logical error

        # New errors appearing?

        self.visible_state = np.copy(self.next_visible_state)

        return self.next_visible_state, reward, done
    

    def _is_logically_correct(self):
        """
        Returns True if the current hidden_state does NOT contain
        a logical X or logical Z error.

        hidden_state[:,:,0] = X-component  (+1 or -1)
        hidden_state[:,:,1] = Z-component  (+1 or -1)
        """

        hx = self.hidden_state[:, :, 0]   # X (bit) errors indicate Z-type logical operators
        hz = self.hidden_state[:, :, 1]   # Z (phase) errors indicate X-type logical operators

        d = self.d

        # Check for Logical Z error (vertical chain of X flips) 
        # If any column has odd number of X-errors -> logical Z
        for col in range(d):
            if np.sum(hx[:, col]) == -d:   
                return False                # logical Z occurred

        # Check for Logical X error (horizontal chain of Z flips) 
        # If any row has odd number of Z-errors -> logical X
        for row in range(d):
            if np.sum(hz[row, :]) == -d:   # 
                return False                # logical X occurred

        # If none found -> no logical error
        return True



    def render(self, figsize=8):
        
        # --- Color palette ---
        COLOR_X_PLAQ = "#ffd8a8"        # light orange
        COLOR_Z_PLAQ = "#a5d8ff"        # light blue
        COLOR_SYND_OK = "white"
        COLOR_SYND_Z_BAD = "#b00000"    # bright dark red (for Z syndromes)
        COLOR_SYND_X_BAD = "#7e307e"    # deep purple (for X syndromes)
        COLOR_DATA_OK = "black"
        COLOR_DATA_ERR = "#ffcc00"      # gold
        EDGE_COLOR = "black"            
        X_BOUNDARY_EDGE_COLOR = "#ff8c00" # Orange for X boundary links
        Z_BOUNDARY_EDGE_COLOR = "#007bff" # Blue for Z boundary links
        BG_COLOR = "#d9dedb"            

        L = 2 * self.d + 1  # lattice size in coordinates

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        ax.set_aspect("equal")
        ax.set_title(f"Rotated Surface Code (d={self.d})", fontsize=16, pad=12)

        # coordinate system: we will treat (i,j) as (y,x) when plotting
        ax.set_xlim(-0.6, L - 0.4)
        ax.set_ylim(-0.6, L - 0.4)
        ax.invert_yaxis()      
        ax.axis("off")
        ax.add_patch(Rectangle((-0.6, -0.6), L+0.2, L+0.2, facecolor=BG_COLOR, zorder=0))

        # --- Coordinate Sets and Maps ---
        data_qubit_set = set(map(tuple, self.data_qubits_coord.tolist()))
        stab_coords = np.vstack([self.x_stabs_coord, self.z_stabs_coord])
        stab_set = set(map(tuple, stab_coords.tolist()))
        
        # Map stab coordinates to their type for easy lookup
        stab_type = {}
        for i, j in self.x_stabs_coord:
            stab_type[(i, j)] = 'X'
        for i, j in self.z_stabs_coord:
            stab_type[(i, j)] = 'Z'

        # --------------------------
        # 1) Draw colored plaquettes (Interior squares and boundary triangles)
        # --------------------------
        
        # Z-plaquettes (light blue)
        for (i, j) in self.z_stabs_coord:
            # Check if stabilizer is strictly interior: 2 <= i,j <= 2d-2
            is_interior = (i > 0 and i < 2*self.d and j > 0 and j < 2*self.d)
            if is_interior:
                # Draw full 1x1 square for interior Z stabilizers
                ax.add_patch(Rectangle((j-1, i-1), 2.0, 2.0,
                                    facecolor=COLOR_Z_PLAQ, edgecolor=EDGE_COLOR, linewidth=1.6, 
                                    alpha=0.8,zorder=2))
            else:     
                if j == 0:         # left, pointing right
                    verts = [(j, i), (j+1, i+1), (j+1, i-1)]
                elif j == 2*self.d:       # right, pointing left
                    verts = [(j, i), (j-1, i+1), (j-1, i-1)]

                tri = Polygon(verts, facecolor=COLOR_Z_PLAQ, linewidth=1.6, zorder=2)
                ax.add_patch(tri)
        # X-plaquettes (light orange)
        for (i, j) in self.x_stabs_coord:
            # Check if stabilizer is strictly interior: 2 <= i,j <= 2d-2
            is_interior = (i > 0 and i < 2*self.d and j > 0 and j < 2*self.d)
            if is_interior:
                # Draw full 1x1 square for interior X stabilizers
                ax.add_patch(Rectangle((j-1, i-1), 2.0, 2.0,
                                    facecolor=COLOR_X_PLAQ, edgecolor=EDGE_COLOR, linewidth=1.6, 
                                    alpha=0.8, zorder=2))
            else:
                if i == 0:         # left, pointing right
                    verts = [(j, i), (j+1, i+1), (j-1, i+1)]
                elif i == 2*self.d:       # right, pointing left
                    verts = [(j, i), (j+1, i-1), (j-1, i-1)]

                tri = Polygon(verts, facecolor=COLOR_X_PLAQ, linewidth=1.6, zorder=2)
                ax.add_patch(tri)

        # --------------------------
        # 2) Draw Edges 
        # --------------------------
        
        for (i, j) in data_qubit_set:
            # Edges between adjacent data qubits (Black links)
            # Right neighbor (i, j+2)
            if (i, j+2) in data_qubit_set:
                ax.plot([j, j+2], [i, i], color=EDGE_COLOR, linewidth=1.6, zorder=2)
            # Down neighbor (i+2, j)
            if (i+2, j) in data_qubit_set:
                ax.plot([j, j], [i, i+2], color=EDGE_COLOR, linewidth=1.6, zorder=2)
            
        # --------------------------
        # 3) Draw syndrome nodes (white, red, or purple) at stabilizer centers
        # --------------------------
        # X-stabilizers (Purple for -1)
        for (i, j) in self.x_stabs_coord:
            s = self.syndrome_lattice[i, j, 0] # X Syndrome
            color = COLOR_SYND_X_BAD if s == -1 else COLOR_SYND_OK
            ax.add_patch(Circle((j, i), 0.18, facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.9, zorder=4))

        # Z-stabilizers (Red for -1)
        for (i, j) in self.z_stabs_coord:
            s = self.syndrome_lattice[i, j, 1] # Z Syndrome
            color = COLOR_SYND_Z_BAD if s == -1 else COLOR_SYND_OK
            ax.add_patch(Circle((j, i), 0.18, facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.9, zorder=4))

        # --------------------------
        # 4) Draw data qubits (black or gold)
        # --------------------------
        for (i, j) in self.data_qubits_coord:
            ii = (i - 1) // 2
            jj = (j - 1) // 2
            hx = int(self.hidden_state[ii, jj, 0])  # X indicator: -1 => X present
            hz = int(self.hidden_state[ii, jj, 1])  # Z indicator: -1 => Z present

            if hx == 1 and hz == 1:
                face = COLOR_DATA_OK
                label = None
            else:
                face = COLOR_DATA_ERR
                if hx == -1 and hz == 1:
                    label = "X"
                elif hx == 1 and hz == -1:
                    label = "Z"
                else:
                    label = "Y"

            ax.add_patch(Circle((j, i), 0.16, facecolor=face, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=6))

            if label is not None:
                ax.text(j, i + 0.02, label, ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black", zorder=7)

        # --------------------------
        # 5) Legend 
        # --------------------------
        
        legend_items = [
            Patch(facecolor=COLOR_X_PLAQ, edgecolor=EDGE_COLOR, label="X Plaquette (Orange, Interior)"),
            Patch(facecolor=COLOR_Z_PLAQ, edgecolor=EDGE_COLOR, label="Z Plaquette (Blue, Interior)"),
            Circle((0, 0), 0.12, facecolor=COLOR_SYND_OK, edgecolor=EDGE_COLOR, label="Syndrome = +1 (White)"),
            Circle((0, 0), 0.12, facecolor=COLOR_SYND_Z_BAD, edgecolor=EDGE_COLOR, label="Z Syndrome = -1 (Red)"),
            Circle((0, 0), 0.12, facecolor=COLOR_SYND_X_BAD, edgecolor=EDGE_COLOR, label="X Syndrome = -1 (Purple)"),
            Circle((0, 0), 0.12, facecolor=COLOR_DATA_OK, edgecolor=EDGE_COLOR, label="Data Qubit (No Error)"),
            Circle((0, 0), 0.12, facecolor=COLOR_DATA_ERR, edgecolor=EDGE_COLOR, label="Data Qubit (Error: X/Z/Y)"),
        ]

        ax.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(1.4, 1.0),
                  frameon=True, fontsize=8)
        
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    env = SurfaceCode(
        d = 5,
        p_phys = 0.2
    )
    env.render()
