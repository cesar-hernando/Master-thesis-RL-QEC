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


def _binned_curve(x, y, n_bins=40, min_count=20):
    """Return binned means for noisy binary/shot-level relationships."""
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_min = np.min(x)
    x_max = np.max(x)
    if x_max <= x_min:
        return np.array([x_min]), np.array([np.mean(y)]), np.array([len(y)])

    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers, means, counts = [], [], []

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i < n_bins - 1:
            mask = (x >= left) & (x < right)
        else:
            mask = (x >= left) & (x <= right)

        count = int(np.sum(mask))
        if count < min_count:
            continue

        centers.append((left + right) * 0.5)
        means.append(float(np.mean(y[mask])))
        counts.append(count)

    return np.asarray(centers), np.asarray(means), np.asarray(counts)


def _render_degeneracy_dashboard(degen_data, oracle_ler, output_file="plots/weight_degeneracy_dashboard.html"):
    """Render a compact dashboard that visualizes weight-performance degeneracy."""
    shot_mse = np.asarray(degen_data["shot_weight_mse"], dtype=np.float64)
    shot_err = np.asarray(degen_data["shot_agent_error"], dtype=np.float64)
    shot_match_oracle = np.asarray(degen_data["shot_match_oracle_obs"], dtype=np.float64)
    shot_step = np.asarray(degen_data["shot_step"], dtype=np.float64)

    cp_step = np.asarray(degen_data["checkpoint_step"], dtype=np.float64)
    cp_mse = np.asarray(degen_data["checkpoint_weight_mse"], dtype=np.float64)
    cp_ler = np.asarray(degen_data["checkpoint_test_ler"], dtype=np.float64)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Checkpoint Frontier: Weight MSE vs Test LER",
            "Shot-level Error Probability vs Weight MSE",
            "Shot-level Oracle-Agreement vs Weight MSE",
            "Temporal Evolution: Checkpoint MSE vs LER"
        )
    )

    if cp_mse.size > 0:
        fig.add_trace(
            go.Scatter(
                x=cp_mse,
                y=cp_ler,
                mode="markers",
                marker=dict(size=7, color=cp_step, colorscale="Viridis", showscale=True, colorbar=dict(title="Step")),
                name="Checkpoints"
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[float(np.min(cp_mse)), float(np.max(cp_mse))],
                y=[oracle_ler, oracle_ler],
                mode="lines",
                line=dict(color="green", dash="dash"),
                name="Oracle LER"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(x=cp_step, y=cp_mse, mode="lines+markers", name="Checkpoint Weight MSE"),
            row=2,
            col=2
        )
        fig.add_trace(
            go.Scatter(x=cp_step, y=cp_ler, mode="lines+markers", name="Checkpoint Test LER"),
            row=2,
            col=2
        )

    if shot_mse.size > 0:
        fig.add_trace(
            go.Scatter(
                x=shot_mse,
                y=shot_err,
                mode="markers",
                marker=dict(size=4, color=shot_step, colorscale="Plasma", opacity=0.25),
                name="Shot outcomes"
            ),
            row=1,
            col=2
        )

        bx, by, _ = _binned_curve(shot_mse, shot_err)
        if bx.size > 0:
            fig.add_trace(
                go.Scatter(x=bx, y=by, mode="lines", line=dict(width=3, color="black"), name="Binned error rate"),
                row=1,
                col=2
            )

        fig.add_trace(
            go.Scatter(
                x=shot_mse,
                y=shot_match_oracle,
                mode="markers",
                marker=dict(size=4, color=shot_step, colorscale="Cividis", opacity=0.25),
                name="Oracle agreement"
            ),
            row=2,
            col=1
        )

        bx2, by2, _ = _binned_curve(shot_mse, shot_match_oracle)
        if bx2.size > 0:
            fig.add_trace(
                go.Scatter(x=bx2, y=by2, mode="lines", line=dict(width=3, color="black"), name="Binned agreement"),
                row=2,
                col=1
            )

    fig.update_xaxes(title_text="Weight MSE to Oracle", row=1, col=1)
    fig.update_yaxes(title_text="Test LER", row=1, col=1)

    fig.update_xaxes(title_text="Weight MSE to Oracle", row=1, col=2)
    fig.update_yaxes(title_text="P(logical error)", row=1, col=2, range=[-0.05, 1.05])

    fig.update_xaxes(title_text="Weight MSE to Oracle", row=2, col=1)
    fig.update_yaxes(title_text="P(match oracle pred)", row=2, col=1, range=[-0.05, 1.05])

    fig.update_xaxes(title_text="Global evaluation step", row=2, col=2)
    fig.update_yaxes(title_text="Metric value", row=2, col=2)

    fig.update_layout(
        title="Weight Degeneracy Diagnostics",
        template="plotly_white",
        height=900,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.write_html(output_file)
    print(f"[*] Degeneracy dashboard saved to '{output_file}'.")


def _summarize_degeneracy(degen_data, oracle_ler, ler_tolerance):
    """Print quantitative indicators for how much degeneracy the policy exploits."""
    shot_mse = np.asarray(degen_data["shot_weight_mse"], dtype=np.float64)
    shot_agent_error = np.asarray(degen_data["shot_agent_error"], dtype=np.float64)
    shot_oracle_error = np.asarray(degen_data["shot_oracle_error"], dtype=np.float64)
    shot_match_oracle = np.asarray(degen_data["shot_match_oracle_obs"], dtype=np.float64)

    cp_mse = np.asarray(degen_data["checkpoint_weight_mse"], dtype=np.float64)
    cp_ler = np.asarray(degen_data["checkpoint_test_ler"], dtype=np.float64)

    if shot_mse.size == 0:
        print("[!] No degeneracy statistics available (no evaluated shots after burn-in).")
        return

    high_mse_thr = float(np.quantile(shot_mse, 0.75))
    high_mse_mask = shot_mse >= high_mse_thr

    deg_shot_rate = float(np.mean(high_mse_mask & (shot_match_oracle > 0.5)))
    agent_err_high = float(np.mean(shot_agent_error[high_mse_mask])) if np.any(high_mse_mask) else float("nan")
    oracle_err_high = float(np.mean(shot_oracle_error[high_mse_mask])) if np.any(high_mse_mask) else float("nan")

    if cp_mse.size > 1 and np.std(cp_mse) > 0 and np.std(cp_ler) > 0:
        cp_corr = float(np.corrcoef(cp_mse, cp_ler)[0, 1])
    else:
        cp_corr = float("nan")

    if cp_ler.size > 0:
        near_oracle = cp_ler <= (oracle_ler + ler_tolerance)
        near_oracle_rate = float(np.mean(near_oracle))
        cp_high_thr = float(np.quantile(cp_mse, 0.75))
        cp_high_mask = cp_mse >= cp_high_thr
        high_mse_near_oracle = float(np.mean(cp_high_mask & near_oracle))
    else:
        near_oracle_rate = float("nan")
        high_mse_near_oracle = float("nan")

    print("\n" + "=" * 60)
    print("WEIGHT DEGENERACY SUMMARY")
    print("=" * 60)
    print(f"Shot-level high-MSE threshold (Q75): {high_mse_thr:.6f}")
    print(f"Degeneracy shot rate (high MSE + oracle-agree): {deg_shot_rate:.4f}")
    print(f"Agent error rate on high-MSE shots: {agent_err_high:.6f}")
    print(f"Oracle error rate on high-MSE shots: {oracle_err_high:.6f}")
    print(f"Checkpoint corr(MSE, LER): {cp_corr:.4f}")
    print(f"Near-oracle checkpoint rate (eps={ler_tolerance:g}): {near_oracle_rate:.4f}")
    print(f"High-MSE + near-oracle checkpoint rate: {high_mse_near_oracle:.4f}")


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

    degen_data = {
        "shot_step": [],
        "shot_weight_mse": [],
        "shot_weight_l1": [],
        "shot_agent_error": [],
        "shot_oracle_error": [],
        "shot_match_oracle_obs": [],
        "checkpoint_step": [],
        "checkpoint_weight_mse": [],
        "checkpoint_test_ler": [],
    }
    
    saved_shots = []
    start_time = time.time()
    
    for episode in range(config['test_episodes']):
        obs, info = env.reset(seed=test_seeds[episode])
        done = False
        step_count = 0
        ep_gnn_errs, ep_cma_errs, ep_static_errs = 0, 0, 0

        global_ep_start = episode * n_shots
        degen_data["checkpoint_step"].append(global_ep_start)
        degen_data["checkpoint_weight_mse"].append(float(info["weights_mse_error"]))
        degen_data["checkpoint_test_ler"].append(float(info["initial_test_ler"]))
        
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
                global_step = global_ep_start + step_count + 1
                
                gnn_eval_errors += err_gnn
                cma_eval_errors += err_cma
                static_eval_errors += err_static
                oracle_eval_errors += err_oracle
                
                ep_gnn_errs += err_gnn
                ep_cma_errs += err_cma
                ep_static_errs += err_static

                shot_mse = float(np.mean((w_final - env.oracle_weights) ** 2))
                shot_l1 = float(np.mean(np.abs(w_final - env.oracle_weights)))
                match_oracle_obs = int(info["pred_obs"] == info["oracle_pred_obs"])

                degen_data["shot_step"].append(global_step)
                degen_data["shot_weight_mse"].append(shot_mse)
                degen_data["shot_weight_l1"].append(shot_l1)
                degen_data["shot_agent_error"].append(err_gnn)
                degen_data["shot_oracle_error"].append(err_oracle)
                degen_data["shot_match_oracle_obs"].append(match_oracle_obs)

                if ((step_count + 1) % env.update_period) == 0:
                    degen_data["checkpoint_step"].append(global_step)
                    degen_data["checkpoint_weight_mse"].append(float(info["weights_mse_error"]))
                    degen_data["checkpoint_test_ler"].append(float(info["test_ler"]))

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

    ler_tolerance = config.get('degeneracy_ler_tolerance', 2e-4)
    _summarize_degeneracy(degen_data, ler_oracle, ler_tolerance)
    _render_degeneracy_dashboard(degen_data, ler_oracle)
    
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
        'model_path': 'models/sac_gnn_64.pth',
        'distance': 5,
        'n_rounds': 5,
        'p': 0.004,
        'p_gate_zz': 0.0,
        'mismatch': 1.0,
        'n_shots': 100_000,
        'burn_in_steps': 0,
        'bypass_threshold': 2,
        'action_scale': 5.0,
        'update_period': 1_000_000,
        'prior_shots': 1_000,
        'local_action_only': True,
        'local_action_hops': 1,
        'hidden_dim': 128,
        'lr': 1e-4,
        'gamma': 0.0,
        'tau': 0.005,
        'alpha': 0.01,
        'batch_size': 128,
        'buffer_capacity': 100_000,
        'update_frequency': 100,
        'train_episodes': 50,
        'test_episodes': 5,  # Reduced slightly so you get your LER stats faster
        'degeneracy_ler_tolerance': 2e-4,
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
        memory_type='z', 
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