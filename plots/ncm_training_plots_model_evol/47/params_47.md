HIDDEN_DIM=128
TRAIN_EPISODES=270

echo "Starting Run 47: Gamma=0.0 | Layers=1 | Mismatch=30.0 | action_scale=6 | Using log joint probs| h.d.=128"
python -u run_job.py --mode 0 \
    --gamma 0.0 \
    --n_layers 1 \
    --update_period 1000 \
    --mismatch 30.0 \
    --hidden_dim $HIDDEN_DIM \
    --train_episodes $TRAIN_EPISODES \
    --n_shots 65000 \
    --burn_in_steps 15000 \
    --action_scale 6 \
    --no-use_pearson_correlation \
    --use_log_joint_prob \
    --model_path models/sac_gnn_47.pth \
    --training_metrics_filename training_metrics_47.png