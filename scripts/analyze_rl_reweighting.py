"""
Unified execution script for evaluating Logical Error Rates (LER) 
and visually analyzing specific complex shots where the SAC-GNN agent 
reweights the MWPM decoding graph.
"""

import time
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent


def _render_shot_gallery(env, saved_shots):
    """Builds a Plotly slideshow containing multiple shots with Slider navigation."""
    
    if not saved_shots:
        print("[!] No diverging shots found to render.")
        return

    # Setup subplots
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("<b>Pass 1 (Static MWPM)</b>", "<b>Pass 2 (GNN Reweighted)</b>")
    )

    # Helper: Extract Coords safely
    def get_coord(node):
        if node == -1: return None
        c = env.detector_coords.get(node, None)
        if c is None: return None
        z_coord = float(c[2]) if len(c) > 2 else 0.0
        return [float(c[0]), float(c[1]), z_coord]

    # --- TRACE BUILDER ---
    def build_graph_traces(shot_data, is_pass_1):
        line_x, line_y, line_z, line_colors, line_texts = [], [], [], [], []
        sol_x, sol_y, sol_z, sol_texts = [], [], [], []
        
        syndrome = shot_data['syndrome']
        if hasattr(syndrome, "ndim") and syndrome.ndim == 2:
            fired = set(np.where(syndrome[0] == 1)[0].tolist())
        else:
            fired = set(np.where(syndrome == 1)[0].tolist())

        # Edges
        for i, (u, v) in enumerate(env.dec_edge_list):
            cu, cv = get_coord(u), get_coord(v)
            if cu is None and cv is None: continue
            
            if cu is None: cu = [cv[0], cv[1], cv[2] - 0.5]
            if cv is None: cv = [cu[0], cu[1], cu[2] - 0.5]

            u_lbl = "Boundary" if u == -1 else f"D{u}"
            v_lbl = "Boundary" if v == -1 else f"D{v}"

            if is_pass_1:
                color = "rgba(150, 150, 150, 0.2)"
                w_str = f"{shot_data['w_init'][i]:.3f}"
                hover = f"<b>Edge: {i}</b> ({u_lbl} &mdash; {v_lbl})<br>W: {w_str}"
                is_selected = i in shot_data['p1_idx']
            else:
                d = shot_data['delta'][i]
                mask = shot_data['mask']
                if mask[i] > 0 and abs(d) > 0.01:
                    color = "rgba(0, 100, 255, 0.7)" if d < 0 else "rgba(255, 50, 50, 0.7)"
                else:
                    color = "rgba(150, 150, 150, 0.1)"
                    
                hover = (
                    f"<b>Edge: {i}</b> ({u_lbl} &mdash; {v_lbl})<br>"
                    f"Init W: {shot_data['w_init'][i]:.3f}<br>"
                    f"Action: {shot_data['action'][i]:.3f} (Mask: {int(mask[i])})<br>"
                    f"Delta: <b>{d:+.3f}</b><br>"
                    f"Final W: {shot_data['w_final'][i]:.3f}"
                )
                is_selected = i in shot_data['p2_idx']

            if is_selected:
                sol_x.extend([cu[0], cv[0], None])
                sol_y.extend([cu[1], cv[1], None])
                sol_z.extend([cu[2], cv[2], None])
                sol_texts.extend([hover, hover, ""])
            else:
                line_x.extend([cu[0], cv[0], None])
                line_y.extend([cu[1], cv[1], None])
                line_z.extend([cu[2], cv[2], None])
                line_colors.extend([color, color, color])
                line_texts.extend([hover, hover, ""])
                
        # Nodes
        nx, ny, nz, n_colors, n_texts = [], [], [], [], []
        for n, c in env.detector_coords.items():
            nx.append(float(c[0]))
            ny.append(float(c[1]))
            nz.append(float(c[2]) if len(c) > 2 else 0.0)
            
            is_fired = n in fired
            j, y = int(round(c[0])), int(round(c[1]))
            t = 'Z' if (y % 4 == 0 and j % 4 == 0) or (y % 4 == 2 and j % 4 == 2) else 'X'
            
            if is_fired:
                n_colors.append('#33aaff' if t == 'Z' else '#ff4d4d')
                n_texts.append(f"D{n} ({t}) - FIRED")
            else:
                n_colors.append('#005f87' if t == 'Z' else '#b22222')
                n_texts.append(f"D{n} ({t})")
                
        return line_x, line_y, line_z, line_colors, line_texts, sol_x, sol_y, sol_z, sol_texts, nx, ny, nz, n_colors, n_texts

    # --- POPULATE ALL SHOTS ---
    traces_per_shot = 6  
    
    for i, shot in enumerate(saved_shots):
        is_visible = (i == 0) 
        
        # PASS 1
        lx, ly, lz, lc, lt, sx, sy, sz, st, nx, ny, nz, nc, nt = build_graph_traces(shot, is_pass_1=True)
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(width=2, color=lc), text=lt, hoverinfo='text', name=f'Base {i}', visible=is_visible), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode='lines', line=dict(width=7, color='#d4af37'), text=st, hoverinfo='text', name=f'MWPM {i}', visible=is_visible), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=nx, y=ny, z=nz, mode='markers', marker=dict(size=5, color=nc, line=dict(width=1, color='white')), text=nt, hoverinfo='text', name=f'Dets {i}', visible=is_visible), row=1, col=1)

        # PASS 2
        lx2, ly2, lz2, lc2, lt2, sx2, sy2, sz2, st2, nx2, ny2, nz2, nc2, nt2 = build_graph_traces(shot, is_pass_1=False)
        fig.add_trace(go.Scatter3d(x=lx2, y=ly2, z=lz2, mode='lines', line=dict(width=3, color=lc2), text=lt2, hoverinfo='text', name=f'Reweight {i}', visible=is_visible, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=sx2, y=sy2, z=sz2, mode='lines', line=dict(width=7, color='#d4af37'), text=st2, hoverinfo='text', name=f'GNN Path {i}', visible=is_visible, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=nx2, y=ny2, z=nz2, mode='markers', marker=dict(size=5, color=nc2, line=dict(width=1, color='white')), text=nt2, hoverinfo='text', name=f'Dets {i}', visible=is_visible, showlegend=False), row=1, col=2)

    # --- SLIDER LOGIC (Play Button Removed) ---
    steps = []
    for i, shot in enumerate(saved_shots):
        visibility = [False] * (len(saved_shots) * traces_per_shot)
        for j in range(traces_per_shot):
            visibility[i * traces_per_shot + j] = True
            
        p1_res = 'Success' if shot['p1_corr'] else 'ERROR'
        p2_res = 'Success' if shot['p2_corr'] else 'ERROR'
            
        title_text = (
            f"<b>GNN Decision Inspector (Shot {i+1}/{len(saved_shots)} - Step {shot['step']})</b><br>"
            f"<sup>Pass 1: {p1_res} | Pass 2: {p2_res} | True Obs: {shot['true_obs']}</sup>"
        )
            
        step = dict(
            method="update",
            args=[
                {"visible": visibility},
                {"title.text": title_text}
            ],
            label=f"Shot {i+1}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Viewing: "},
        pad={"t": 50},
        steps=steps
    )]

    # --- FORMATTING ---
    scene_config = dict(
        xaxis_title="X (Space)", yaxis_title="Y (Space)", zaxis_title="T (Time)",
        aspectmode="data", bgcolor='rgb(245, 245, 245)'
    )
    
    init_shot = saved_shots[0]
    init_title = (
        f"<b>GNN Decision Inspector (Shot 1/{len(saved_shots)} - Step {init_shot['step']})</b><br>"
        f"<sup>Pass 1: {'Success' if init_shot['p1_corr'] else 'ERROR'} | Pass 2: {'Success' if init_shot['p2_corr'] else 'ERROR'} | True Obs: {init_shot['true_obs']}</sup>"
    )
    
    fig.update_layout(
        title=dict(
            text=init_title,
            x=0.5, y=0.95, xanchor='center', yanchor='top'
        ),
        scene=scene_config, scene2=scene_config,
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05),
        margin=dict(l=0, r=0, b=0, t=100),
        sliders=sliders
        # REMOVED the broken updatemenus (Play Button)
    )

    os.makedirs('plots', exist_ok=True)
    filename = "plots/shot_analysis_gallery.html"
    fig.write_html(filename)
    print(f"\n[*] Render complete! Found {len(saved_shots)} diverging shots. Open '{filename}'.")


def evaluate_ler_and_extract_gallery(env, agent, config, max_shots=10):
    """
    Evaluates the Logical Error Rate (LER) over all test episodes (excluding burn-in),
    and simultaneously harvests complex, diverging shots for the Plotly gallery.
    """
    print(f"\n{'='*60}")
    print(f"STARTING UNIFIED EVALUATION & GALLERY HARVESTING")
    print(f"{'='*60}")
    
    agent.load_models(config['model_path'])
    bypass_threshold = config.get('bypass_threshold', 2)

    test_seeds = [int(np.random.randint(0, 1_000_000)) for _ in range(config['test_episodes'])]
    
    n_shots = config['n_shots']
    burn_in_steps = config.get('burn_in_steps', 0)
    
    # Calculate exactly how many shots are actually evaluated
    eval_shots_per_ep = n_shots - burn_in_steps
    total_eval_shots = config['test_episodes'] * eval_shots_per_ep
    
    print(f"Total Shots per Episode: {n_shots}")
    print(f"Burn-in Shots skipped:   {burn_in_steps}")
    print(f"Active Evaluation Shots: {eval_shots_per_ep} per episode")
    print(f"Total Evaluation Shots:  {total_eval_shots} across {config['test_episodes']} episodes\n")

    gnn_eval_errors = 0
    cma_eval_errors = 0     # <--- NEW: Adaptive First-Pass (Zero Action)
    static_eval_errors = 0
    oracle_eval_errors = 0
    
    saved_shots = []
    start_time = time.time()
    
    for episode in range(config['test_episodes']):
        obs, info = env.reset(seed=test_seeds[episode])
        done = False
        step_count = 0
        ep_gnn_errs, ep_cma_errs, ep_static_errs = 0, 0, 0
        
        while not done:
            n_flashes = np.sum(env.current_syndrome != 0)
            
            # --- CAPTURE PRE-ACTION STATE ---
            syndrome = env.current_syndrome
            true_obs = env.current_true_obs
            pass_1_obs = env.current_first_pass_pred_obs
            pass_1_selected_idx = set(env.current_first_pass_selected_idx.tolist()) if env.current_first_pass_selected_idx is not None else set()
            w_initial = obs['node_features'][:, 0].copy()
            mask = obs['action_mask']

            # --- 1. ACTION LOGIC ---
            if step_count < burn_in_steps:
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
                applied_delta = np.zeros_like(action)
            else:
                if n_flashes <= bypass_threshold:
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                else:
                    action = agent.select_action(obs, evaluate=True)
                applied_delta = action * mask * env.action_scale

            w_final = np.clip(w_initial + applied_delta, env.min_weight, env.max_weight)
            
            # --- 2. STEP ENVIRONMENT ---
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # --- 3. ERROR TRACKING (ONLY AFTER BURN-IN) ---
            if step_count >= burn_in_steps:
                err_gnn = int(info["logical_error"])
                err_cma = int(info["first_pass_obs"] != info["true_obs"]) # <--- NEW: CMA Error check
                err_static = int(info["static_pred_obs"] != info["true_obs"])
                err_oracle = int(info["oracle_pred_obs"] != info["true_obs"])
                
                gnn_eval_errors += err_gnn
                cma_eval_errors += err_cma
                static_eval_errors += err_static
                oracle_eval_errors += err_oracle
                
                ep_gnn_errs += err_gnn
                ep_cma_errs += err_cma
                ep_static_errs += err_static

                # --- 4. GALLERY HARVESTING ---
                pass_2_selected_idx = set(info['selected_edges_second_pass_idx'].tolist())
                pass_2_obs = info['pred_obs']
                
                # --- 4. GALLERY HARVESTING ---
                pass_2_selected_idx = set(info['selected_edges_second_pass_idx'].tolist())
                pass_2_obs = info['pred_obs']
                
                # Filter: Did the GNN change the LOGICAL PREDICTION? Is it complex enough? Do we have space?
                if (pass_1_obs != pass_2_obs) and (len(saved_shots) < max_shots):
                    saved_shots.append({
                        'step': env.step_count,
                        'syndrome': syndrome,
                        'true_obs': true_obs,
                        'w_init': w_initial,
                        'w_final': w_final,
                        'delta': applied_delta,
                        'action': action,
                        'mask': mask,
                        'p1_idx': pass_1_selected_idx,
                        'p2_idx': pass_2_selected_idx,
                        'p1_corr': (true_obs == pass_1_obs),
                        'p2_corr': (true_obs == pass_2_obs)
                    })
                    print(f"  -> Captured logically diverging shot at Ep {episode+1}, Step {env.step_count} ({len(saved_shots)}/{max_shots})")

            obs = next_obs
            step_count += 1
            
        print(f"  Test Ep {episode+1:02d} | GNN Errors: {ep_gnn_errs} | CMA Errors: {ep_cma_errs} | Static Errors: {ep_static_errs}")

    # --- FINAL MATH & PRINTING ---
    ler_gnn = gnn_eval_errors / total_eval_shots
    ler_cma = cma_eval_errors / total_eval_shots
    ler_static = static_eval_errors / total_eval_shots
    ler_oracle = oracle_eval_errors / total_eval_shots
    
    print(f"\n{'='*60}")
    print(f"FINAL LOGICAL ERROR RATES (LER)")
    print(f"{'='*60}")
    print(f"Static MWPM (No Drift Adapt): {ler_static:.6f}  ({static_eval_errors}/{total_eval_shots})")
    print(f"CMA-Only (Adaptive Pass 1):   {ler_cma:.6f}  ({cma_eval_errors}/{total_eval_shots})")
    print(f"GNN Agent (Adaptive Pass 2):  {ler_gnn:.6f}  ({gnn_eval_errors}/{total_eval_shots})")
    print(f"Oracle (Perfect Knowledge):   {ler_oracle:.6f}  ({oracle_eval_errors}/{total_eval_shots})")
    
    if ler_static > 0:
        improvement_static = ((ler_static - ler_gnn) / ler_static) * 100
        print(f"\n-> GNN Improvement over Static:   {improvement_static:+.2f}%")
    if ler_cma > 0:
        improvement_cma = ((ler_cma - ler_gnn) / ler_cma) * 100
        print(f"-> GNN Improvement over CMA-Only: {improvement_cma:+.2f}%")
        
    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds.")
    
    # --- RENDER GALLERY ---
    if saved_shots:
        print("\nBuilding interactive gallery...")
        _render_shot_gallery(env, saved_shots)
    else:
        print("\n[!] No complex diverging shots were found to generate a gallery.")


if __name__ == "__main__":
    ######################################
    # 1. HYPERPARAMETERS & CONFIGURATION #
    ######################################
    CONFIG = {
        'model_path': 'models/sac_gnn_19.pth',
        'distance': 5,
        'n_rounds': 5,
        'p': 0.004,
        'p_gate_zz': 0.0,
        'mismatch': 1.0,
        'n_shots': 20_000,
        'burn_in_steps': 0,
        'bypass_threshold': 2,
        'action_scale': 3.0,
        'update_period': 1000,
        'prior_shots': 1000,
        'oracle_reward_coef': 0.0,
        'local_action_only': True,
        'local_action_hops': 1,
        'hidden_dim': 128,
        'lr': 1e-4,
        'gamma': 0.0,
        'tau': 0.005,
        'alpha': 0.01,
        'batch_size': 64,
        'buffer_capacity': 100_000,
        'update_frequency': 100,
        'train_episodes': 50,
        'test_episodes': 5  # Reduced slightly so you get your LER stats faster
    }

    print("Initializing environment...")
    generator = SyndromeDataGenerator(
        distance=CONFIG['distance'], 
        n_rounds=CONFIG['n_rounds'], 
        mismatch=CONFIG['mismatch'],  
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": CONFIG["p"],
            "before_measure_flip_probability": CONFIG["p"],
            "after_reset_flip_probability": CONFIG["p"],
            "before_round_data_depolarization": CONFIG["p"],
            "p_gate_zz": CONFIG["p_gate_zz"]
        }, 
        memory_type='x', 
        n_shots=CONFIG['n_shots'], 
        qec_code='surface_code'
    )

    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=CONFIG['local_action_only'],
        local_action_hops=CONFIG['local_action_hops'],
        action_scale=CONFIG['action_scale'],
        update_period=CONFIG['update_period'],
        prior_shots=CONFIG['prior_shots'],            
        oracle_reward_coef=CONFIG['oracle_reward_coef'], 
        use_pearson_correlation=True,
        use_syndrome_features=False, 
        update_with='DGR',
    )

    sample_obs, _ = env.reset()
    NODE_DIM = sample_obs["node_features"].shape[1]
    
    agent = SACAgent(
        node_dim=NODE_DIM, 
        hidden_dim=CONFIG['hidden_dim'],
        lr=CONFIG['lr'],
        gamma=CONFIG['gamma'],
        tau=CONFIG['tau'],
        alpha=CONFIG['alpha']
    )

    ############################
    # 3. UNIFIED EXECUTION     #
    ############################
    
    # Run evaluation and build gallery simultaneously
    evaluate_ler_and_extract_gallery(
        env, 
        agent, 
        CONFIG,  
        max_shots=10
    )