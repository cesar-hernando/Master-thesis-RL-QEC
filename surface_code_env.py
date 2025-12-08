'''
This module contains the implementation of the surface code's environment
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Patch
import gymnasium as gym

class SurfaceCodeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, d, p_phys, p_meas=0, error_model='X', volume_depth=1, include_masks=False, max_n_steps=100):
        """
        Initialize the rotated surface code environment.

        Parameters
        ----------
        d : int
            Code distance (must be odd)
        p_phys : float
            Physical error probability for data qubits
        p_meas : float
            Measurement error probability for stabilizers
        error_model : str
            Error model to use ('X', 'Z', or 'depolarizing')
        volume_depth : int
            Depth of the syndrome measurement volume
        include_masks : bool
            Whether to include masks in the observation space
        max_n_steps : int
            Maximum number of steps per episode
        """

        super().__init__()

        if d % 2 == 0:
            raise ValueError("The code distance for the rotated surface code must be odd.")
        
        self.d = d
        self.p_phys = p_phys
        self.p_meas = p_meas
        self.error_model = error_model
        self.volume_depth = volume_depth
        self.include_masks = include_masks
        self.max_n_steps = max_n_steps

        # Determine number of actions and observation channels
        if error_model == 'depolarizing':
            self.num_actions = d*d*2 + 1
            if include_masks:
                n_channels = 7
            else:
                n_channels = 4
        elif error_model in ['X', 'Z']:
            self.num_actions = d*d + 1
            if include_masks:
                n_channels = 4
            else:
                n_channels = 2

        # Define the action and observation space
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=-1, 
            high=1,
            shape=(2*d+1,2*d+1,n_channels)
        )

        # Initialize render variables
        self._render_fig = None
        self._render_ax = None

        # Initialize the environment
        self._initialize_environment()


    def _initialize_environment(self):
        """
        Set up the initial state of the surface code environment.
        """

        # Assign coordinates to data qubits and stabilizers
        self.data_qubits_coord, self.x_stabs_coord, self.z_stabs_coord = self._assign_qubit_coordinates()

        # Create masks for data qubits and stabilizers
        self.data_mask, self.x_mask, self.z_mask = self._create_masks()

        # Simulate initial errors
        self.hidden_state, self.syndrome_lattice = self._simulate_errors()

        # Initialize action history
        self.action_history = np.zeros((2*self.d+1, 2*self.d+1, 2)) 
             
        # Stack syndrome and history into visible state
        self._stack_syndrome_and_history()     

        # Initialize hidden syndrome lattice
        self.hidden_syndrome_lattice = self.syndrome_lattice.copy()

        # Initialize cumulative reward and step counter
        self.cumulative_reward = 0
        self.n_steps = 0


    def _assign_qubit_coordinates(self):
        """
        Assign coordinates to data qubits and stabilizers in the rotated surface code.
        
        Returns
        -------
        data_qubits_coord : np.ndarray
            Coordinates of data qubits
        x_stabs_coord : np.ndarray
            Coordinates of X stabilizers
        z_stabs_coord : np.ndarray
            Coordinates of Z stabilizers
        """

        # Assign coordinates to data qubits
        data_qubits_coord = [(i,j) for i in range(1, 2*self.d, 2) for j in range(1, 2*self.d, 2)]

        # Assign coordinates to stabilizers
        x_stabs_coord = []
        z_stabs_coord = []
        for i in range(0, 2*self.d+1, 2):
            for j in range(0, 2*self.d+1, 2):
                if (i % (2*self.d) == 0) or (j % (2*self.d) == 0):
                    # Boundary logic
                    z_stab_left_cond = (j==0) and (i%4==0) and (i!=0)
                    z_stab_right_cond = (j==2*self.d) and ((i+2)%4==0) and (i!=2*self.d)
                    if z_stab_left_cond or z_stab_right_cond:
                        z_stabs_coord.append((i,j))

                    x_stab_top_cond = (i==0) and ((j+2)%4==0) and (j!=2*self.d)
                    x_stab_bottom_cond = (i==2*self.d) and (j%4==0) and (j!=0)
                    if x_stab_top_cond or x_stab_bottom_cond:
                        x_stabs_coord.append((i,j))

                else:
                    # Interior logic
                    if (i/2+j/2) % 2 == 0:
                        z_stabs_coord.append((i,j))
                    else:
                        x_stabs_coord.append((i,j))

        return np.array(data_qubits_coord), np.array(x_stabs_coord), np.array(z_stabs_coord)


    def _create_masks(self):
        """
        Create masks for data qubits and stabilizers.

        Returns
        -------
        data_mask : np.ndarray
            Mask for data qubits
        x_mask : np.ndarray
            Mask for X stabilizers
        z_mask : np.ndarray
            Mask for Z stabilizers
        """
        data_mask = np.zeros((2*self.d+1, 2*self.d+1))
        x_mask = np.zeros((2*self.d+1, 2*self.d+1))
        z_mask = np.zeros((2*self.d+1, 2*self.d+1))

        data_mask[1:2*self.d:2, 1:2*self.d:2] = 1
        x_mask[self.x_stabs_coord[:, 0], self.x_stabs_coord[:, 1]] = 1
        z_mask[self.z_stabs_coord[:, 0], self.z_stabs_coord[:, 1]] = 1
                
        return data_mask, x_mask, z_mask
    

    def _simulate_errors(self):
        """
        Simulate physical errors on data qubits and generate the initial syndrome lattice.
        
        Returns
        -------
        hidden_state : np.ndarray
            The hidden state of the surface code (data qubit errors)
        syndrome_lattice : np.ndarray
            The syndrome lattice indicating stabilizer measurements
        """
 
        syndrome_lattice_x = np.zeros((2*self.d+1, 2*self.d+1))
        syndrome_lattice_z = np.zeros((2*self.d+1, 2*self.d+1))

        # Simulate errors based on the specified error model
        if self.error_model == 'X':
            hidden_state_x = np.random.choice([-1,1], size=(self.d, self.d), p=[self.p_phys,1-self.p_phys])
            hidden_state_z = np.ones((self.d, self.d))
            
            # In the bit flip model, X stabilizers are not triggered 
            syndrome_lattice_x[self.x_stabs_coord[:, 0], self.x_stabs_coord[:, 1]] = 1

            # Trigger Z stabilizers if an odd number of support data qubits are bit-flipped
            for i, j in self.z_stabs_coord:
                support = self._obtain_support_qubits(i, j)
                syndrome_lattice_z[i,j] = np.prod(hidden_state_x[support[:,0], support[:,1]])

        elif self.error_model == 'Z':
            hidden_state_z = np.random.choice([-1,1], size=(self.d, self.d), p=[self.p_phys,1-self.p_phys])
            hidden_state_x = np.ones((self.d, self.d))
            
            # In the phase flip model, Z stabilizers are not triggered 
            syndrome_lattice_z[self.z_stabs_coord[:, 0], self.z_stabs_coord[:, 1]] = 1

            # Trigger X stabilizers if an odd number of support data qubits are phase-flipped
            for i, j in self.x_stabs_coord:
                support = self._obtain_support_qubits(i, j)
                syndrome_lattice_x[i,j] = np.prod(hidden_state_z[support[:,0], support[:,1]])

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
                syndrome_lattice_x[i,j] = np.prod(hidden_state_z[support[:,0], support[:,1]])

            # Trigger Z stabilizers if an odd number of support data qubits are bit-flipped
            for i, j in self.z_stabs_coord:
                support = self._obtain_support_qubits(i, j)
                syndrome_lattice_z[i,j] = np.prod(hidden_state_x[support[:,0], support[:,1]])
        
        # Stack the hidden states and syndrome lattices into single tensors
        hidden_state = np.stack([hidden_state_x, hidden_state_z], axis=-1)
        syndrome_lattice = np.stack([syndrome_lattice_x, syndrome_lattice_z], axis=-1)

        return hidden_state, syndrome_lattice
    

    def _obtain_support_qubits(self, i, j):
        """
        Obtain the coordinates of data qubits that support a given stabilizer.

        Parameters
        ----------
        i : int
            Row coordinate of the stabilizer
        j : int
            Column coordinate of the stabilizer

        Returns
        -------
        support : np.ndarray
            Coordinates of supporting data qubits
        """

        # Determine support based on stabilizer position
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
    

    def _stack_syndrome_and_history(self):
        """
        Stack the syndrome lattice and action history into the visible state.
        """

        if self.error_model == 'depolarizing':
            if self.include_masks:
                self.visible_state = np.stack([self.x_mask, self.z_mask, self.syndrome_lattice[:,:,0], 
                                            self.syndrome_lattice[:,:,1],self.data_mask, 
                                            self.action_history[:,:,0], self.action_history[:,:,1]], axis=-1)
            else:
                self.visible_state = np.stack([self.syndrome_lattice[:,:,0], self.syndrome_lattice[:,:,1], 
                                        self.action_history[:,:,0], self.action_history[:,:,1]], axis=-1)
                
        elif self.error_model == 'X':
            if self.include_masks:
                self.visible_state = np.stack([self.z_mask, self.syndrome_lattice[:,:,1], 
                                            self.data_mask, self.action_history[:,:,0]], axis=-1)
            else:
                self.visible_state = np.stack([self.syndrome_lattice[:,:,1], self.action_history[:,:,0]], axis=-1)

        elif self.error_model == 'Z':
            if self.include_masks:
                self.visible_state = np.stack([self.x_mask, self.syndrome_lattice[:,:,0], 
                                            self.data_mask, self.action_history[:,:,1]], axis=-1)
            else:
                self.visible_state = np.stack([self.syndrome_lattice[:,:,0], self.action_history[:,:,1]], axis=-1)


    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.

        Parameters
        ----------
        seed : int or None
            Seed for the random number generator
        options : dict or None
            Additional options for resetting the environment

        Returns
        -------
        observation (self.visible_state) : np.ndarray
            The initial observation of the environment
        """
        super().reset(seed=seed)
        
        # Re-initialize the environment
        self.hidden_state, self.syndrome_lattice = self._simulate_errors()
        self.action_history = np.zeros((2*self.d+1, 2*self.d+1, 2))
        self._stack_syndrome_and_history()
        self.hidden_syndrome_lattice = self.syndrome_lattice.copy()
        self.n_steps = 0
        self.cumulative_reward = 0
        
        return self.visible_state, {}
    
        
    def step(self, action):
        """
        Take an action in the environment and return the result.

        Parameters
        ----------
        action : int
            The action to take

        Returns
        -------
        observation (self.visible_state) : np.ndarray
            The observation after taking the action (next visible state)
        reward : float
            The reward received after taking the action
        terminated : bool
            Whether the episode has ended due to success (no syndromes and no logical error),
            failure (logical error), or invalid action or repeated action
        truncated : bool
            Whether the episode has reached the maximum number of steps
        info : dict
            Additional information about the step (empty)
        """
        
        reward = 0
        terminated = False
        truncated = False

        # Decode action from integer to array
        action = self._decode_action(action)
        
        # Update action history unless action is the identity or repeated
        if action[2] == 2:
            # Identity action
            reward -= 50
            terminated = True

        elif action[2] == 0:
            # Update X channel
            if int(self.action_history[int(2*action[0]+1), int(2*action[1]+1), 0]) == 0:
                self.action_history[int(2*action[0]+1), int(2*action[1]+1), 0] = 1
                self.hidden_state[int(action[0]), int(action[1]), 0] *= -1
                self._update_hidden_syndrome_lattice(action)  
                # Discount a little bit in every step to make the agent efficient
                reward -= -0.01
            else:
                # Discount and finish episode if action is repeated
                reward -= 30
                terminated = True

        elif action[2] == 1:
            # Update Z channel
            if int(self.action_history[int(2*action[0]+1), int(2*action[1]+1), 1]) == 0:
                self.action_history[int(2*action[0]+1), int(2*action[1]+1), 1] = 1
                self.hidden_state[int(action[0]), int(action[1]), 1] *= -1
                self._update_hidden_syndrome_lattice(action)
                # Discount a little bit in every step to make the agent efficient
                reward -= 0.01
            else:
                # Discount and finish episode if action is repeated
                reward -= 30
                terminated = True

             
        if not terminated:    
            # Reward if all syndromes are +1 and no logical error
            logical_error = self._detect_logical_error()
            if np.all(self.hidden_syndrome_lattice) == 1 and not(logical_error):
                reward += 150
        
            elif logical_error:
                reward -= 100
                terminated = True

            self.cumulative_reward += reward
            # New errors appearing? For now, just static case

        # Increase by one the number of steps and check if the episode should be truncated 
        self.n_steps += 1
        if self.n_steps == self.max_n_steps:
            truncated = True

        # Update the visible state
        if not(terminated) and not(truncated):
            self._stack_syndrome_and_history()
     
        return self.visible_state, reward, terminated, truncated, {}
    

    def _decode_action(self, action):
        """
        Decode an integer action into its corresponding (i, j, t) representation.
            i : row index of the data qubit
            j : column index of the data qubit
            t : type of Pauli operation (0 = X, 1 = Z, 2 = identity)      
        """
        # total number of non-identity actions
        non_id_actions = self.d * self.d * 2

        # identity is the last action
        if action == non_id_actions:
            return None, None, 2  # 2 = identity

        # decode real actions
        t = action % 2          # 0 = X, 1 = Z
        action //= 2
        j = action % self.d
        i = action // self.d

        action = np.array([i, j, t])

        return action

    def _update_hidden_syndrome_lattice(self, action):
        """
        Update the hidden syndrome lattice based on the action taken.
        """
        
        coords_action = np.array([2*action[0]+1, 2*action[1]+1])
        
        candidate_support_stabs = [coords_action + np.array((i,j)) for i in [+1,-1] for j in [+1,-1]]
        candidate_support_stabs = np.array(candidate_support_stabs)
        if action[2] == 0:
            # Check which of the 4 coordinates are Z stabilizers
            is_z_stab = self.z_mask[candidate_support_stabs[:,0], candidate_support_stabs[:,1]] == 1

            # Extract only the valid Z stabilizer coordinates
            support_z_stabs = candidate_support_stabs[is_z_stab]

            # Flip those qubits
            for (x, y) in support_z_stabs:
                self.hidden_syndrome_lattice[x,y,1] *= -1
        
        elif action[2] == 1:
            # Check which of the 4 coordinates are X stabilizers
            is_x_stab = self.x_mask[candidate_support_stabs[:,0], candidate_support_stabs[:,1]] == 1

            # Extract only the valid X stabilizer coordinates
            support_x_stabs = candidate_support_stabs[is_x_stab]

            # Flip those qubits
            for (x, y) in support_x_stabs:
                self.hidden_syndrome_lattice[x,y,0] *= -1


    def _detect_logical_error(self):
        """
        Returns True if the current hidden_state contains
        a logical X or logical Z error.

        hidden_state[:,:,0] = X-component  (+1 or -1)
        hidden_state[:,:,1] = Z-component  (+1 or -1)
        """

        hx = self.hidden_state[:, :, 0]   # X (bit) errors indicate Z-type logical operators
        hz = self.hidden_state[:, :, 1]   # Z (phase) errors indicate X-type logical operators

        d = self.d

        # Check for Logical X error (vertical chain of X flips) 
        # If any column has odd number of X-errors -> logical X
        for col in range(d):
            if np.sum(hx[:, col]) == -d:   
                return True   

        # Check for Logical Z error (horizontal chain of Z flips) 
        # If any row has odd number of Z-errors -> logical Z
        for row in range(d):
            if np.sum(hz[row, :]) == -d:
                return True     

        # If none found -> no logical error
        return False


    def render(self, mode="human", wait_time=None, figsize=(13, 7), play_mode=False):
        """
        Render the surface code lattice in real-time.

        Parameters
        ----------
        mode : str
            Gym render mode (currently only "human" is supported)
        wait_time : float or None
            Time in seconds to wait after rendering. If None, uses render_fps in metadata.
        figsize : int
            Size of the figure
        """

        if play_mode:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal")
            ax.set_title(f"Rotated Surface Code (d={self.d})", fontsize=16, pad=12)
        else:
            # Initialize figure and axes once
            if not hasattr(self, "_render_fig") or self._render_fig is None:
                self._render_fig, self._render_ax = plt.subplots(figsize=figsize)
                self._render_ax.set_aspect("equal")
                self._render_ax.axis("off")
            ax = self._render_ax
            ax.clear()

        # Color palette
        COLOR_X_PLAQ = "#ffd8a8"
        COLOR_Z_PLAQ = "#a5d8ff"
        COLOR_SYND_OK = "white"
        COLOR_SYND_Z_BAD = "#b00000"
        COLOR_SYND_X_BAD = "#7e307e"
        COLOR_DATA_OK = "black"
        COLOR_DATA_ERR = "#ffcc00"
        EDGE_COLOR = "black"
        BG_COLOR = "#d9dedb"

        L = 2 * self.d + 1

        # Background
        ax.set_xlim(-0.6, L - 0.4)
        ax.set_ylim(-0.6, L - 0.4)
        ax.invert_yaxis()
        ax.add_patch(Rectangle((-0.6, -0.6), L+0.2, L+0.2, facecolor=BG_COLOR, zorder=0))

        data_qubit_set = set(map(tuple, self.data_qubits_coord.tolist()))

        ###########################
        # 1) Draw plaquettes      #
        ###########################
        for (i, j) in self.z_stabs_coord:
            is_interior = (i > 0 and i < 2*self.d and j > 0 and j < 2*self.d)
            if is_interior:
                ax.add_patch(Rectangle((j-1, i-1), 2, 2, facecolor=COLOR_Z_PLAQ, edgecolor=EDGE_COLOR, linewidth=1.6, alpha=0.8, zorder=2))
            else:
                if j == 0:
                    verts = [(j, i), (j+1, i+1), (j+1, i-1)]
                elif j == 2*self.d:
                    verts = [(j, i), (j-1, i+1), (j-1, i-1)]
                else:
                    continue
                ax.add_patch(Polygon(verts, facecolor=COLOR_Z_PLAQ, linewidth=1.6, zorder=2))

        for (i, j) in self.x_stabs_coord:
            is_interior = (i > 0 and i < 2*self.d and j > 0 and j < 2*self.d)
            if is_interior:
                ax.add_patch(Rectangle((j-1, i-1), 2, 2, facecolor=COLOR_X_PLAQ, edgecolor=EDGE_COLOR, linewidth=1.6, alpha=0.8, zorder=2))
            else:
                if i == 0:
                    verts = [(j, i), (j+1, i+1), (j-1, i+1)]
                elif i == 2*self.d:
                    verts = [(j, i), (j+1, i-1), (j-1, i-1)]
                else:
                    continue
                ax.add_patch(Polygon(verts, facecolor=COLOR_X_PLAQ, linewidth=1.6, zorder=2))

        ###########################
        # 2) Draw edges           #
        ###########################
        for (i, j) in data_qubit_set:
            if (i, j+2) in data_qubit_set:
                ax.plot([j, j+2], [i, i], color=EDGE_COLOR, linewidth=1.6, zorder=2)
            if (i+2, j) in data_qubit_set:
                ax.plot([j, j], [i, i+2], color=EDGE_COLOR, linewidth=1.6, zorder=2)

        ###########################
        # 3) Draw syndrome nodes  #
        ###########################
        for (i, j) in self.x_stabs_coord:
            s = self.hidden_syndrome_lattice[i, j, 0]
            color = COLOR_SYND_X_BAD if s == -1 else COLOR_SYND_OK
            ax.add_patch(Circle((j, i), 0.18, facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.9, zorder=4))

        for (i, j) in self.z_stabs_coord:
            s = self.hidden_syndrome_lattice[i, j, 1]
            color = COLOR_SYND_Z_BAD if s == -1 else COLOR_SYND_OK
            ax.add_patch(Circle((j, i), 0.18, facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.9, zorder=4))

        ###########################
        # 4) Draw data qubits     #
        ###########################
        for (i, j) in self.data_qubits_coord:
            ii, jj = (i-1)//2, (j-1)//2
            hx = int(self.hidden_state[ii, jj, 0])
            hz = int(self.hidden_state[ii, jj, 1])

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
                ax.text(j, i + 0.02, label, ha="center", va="center", fontsize=10, 
                        fontweight="bold", color="black", zorder=7)

        ###########################
        # 5) Legend               #
        ###########################
        legend_items = [
            Patch(facecolor=COLOR_X_PLAQ, edgecolor=EDGE_COLOR, label="X Plaquette (Orange, Interior)"),
            Patch(facecolor=COLOR_Z_PLAQ, edgecolor=EDGE_COLOR, label="Z Plaquette (Blue, Interior)"),
            Circle((0, 0), 0.12, facecolor=COLOR_SYND_OK, edgecolor=EDGE_COLOR, label="Syndrome = +1"),
            Circle((0, 0), 0.12, facecolor=COLOR_SYND_Z_BAD, edgecolor=EDGE_COLOR, label="Z Syndrome = -1"),
            Circle((0, 0), 0.12, facecolor=COLOR_SYND_X_BAD, edgecolor=EDGE_COLOR, label="X Syndrome = -1"),
            Circle((0, 0), 0.12, facecolor=COLOR_DATA_OK, edgecolor=EDGE_COLOR, label="Data Qubit OK"),
            Circle((0, 0), 0.12, facecolor=COLOR_DATA_ERR, edgecolor=EDGE_COLOR, label="Data Qubit Error"),
        ]
        ax.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(1.4, 1.0), frameon=True, fontsize=8)

        ###########################
        # 6) All-corrected message#
        ###########################
        if np.all(self.hidden_state == 1):
            ax.text(L/2, -L/2, "ALL ERRORS CORRECTED!", ha="center", va="center", fontsize=20, 
                    fontweight="bold", color="lime", bbox=dict(facecolor="black", edgecolor="none", 
                    boxstyle="round,pad=0.4", alpha=0.85), zorder=1000)

        ###########################
        # 7) Show and pause       #
        ###########################

        plt.tight_layout()
        if play_mode:
            plt.show()
        else:
            self._render_fig.canvas.draw()
            self._render_fig.canvas.flush_events()

            if wait_time is None:
                wait_time = 1.0 / self.metadata.get("render_fps", 2)
            elif wait_time != 0:
                plt.pause(wait_time)


    def _encode_action(self, i, j, t):
        """
        Encode a qubit action into a single integer for the environment.

        Parameters
        ----------
        i : int
            Row index of the data qubit (0 <= i < d)
        j : int
            Column index of the data qubit (0 <= j < d)
        t : int
            Action type:
                0 = X
                1 = Z
                2 = Identity

        Returns
        -------
        action : int
            Integer representing the action in the environment's Discrete space
        """
        # Identity is the last action
        if t == 2:
            return self.d * self.d * 2

        # Compute integer action
        action = i * self.d * 2 + j * 2 + t

        return action




if __name__ == '__main__':
    env = SurfaceCodeEnv(
        d = 5,
        p_phys = 0.1,
        error_model='depolarizing',
        include_masks=False
    )
    env.render(play_mode=True)
    done = False
    for _ in range(30):
        print("\nEnter an action in the format: i j type")
        print("Example: 1 1 1  (X correction on qubit (1,1))")
        print("         2 3 2  (Z correction on qubit (2,3))")

        # Ask the user for an action
        user_input = input("Action: ")

        # Convert to an array
        i, j, t = list(map(int, user_input.split()))
        action_int = env._encode_action(i, j, t)
        next_state, reward, done, _ = env.step(action_int)
        if done:
            break
        print(f"\n Current reward = {reward}. Cumulative reward = {env.cumulative_reward}")
        env.render(play_mode=True)

