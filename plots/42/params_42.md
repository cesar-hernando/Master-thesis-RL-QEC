HIDDEN_DIM=128
TRAIN_EPISODES=270

echo "Starting Run 42: Gamma=0.0 | Layers=1 | Mismatch=30.0"
python run_job.py --mode 0 \
    --gamma 0.0 \
    --n_layers 1 \
    --update_period 100 \
    --mismatch 30.0 \
    --hidden_dim $HIDDEN_DIM \
    --train_episodes $TRAIN_EPISODES \
    --n_shots 65000 \
    --burn_in_steps 15000 \
    --model_path models/sac_gnn_42.pth \
    --training_metrics_filename training_metrics_42.png