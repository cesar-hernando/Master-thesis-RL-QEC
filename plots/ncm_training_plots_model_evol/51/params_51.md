HIDDEN_DIM=256
TRAIN_EPISODES=350

echo "Starting Run 51: No burn-in | Gamma=0.99 | Layers=1 | Mismatch=30.0 | action_scale=5 | Using P.C."
python run_job.py --mode 0 \
    --gamma 0.99 \
    --n_layers 1 \
    --update_period 100 \
    --mismatch 30.0 \
    --hidden_dim $HIDDEN_DIM \
    --train_episodes $TRAIN_EPISODES \
    --n_shots 50000 \
    --burn_in_steps 0 \
    --action_scale 5 \
    --use_pearson_correlation \
    --model_path models/sac_gnn_51.pth \
    --training_metrics_filename training_metrics_51.png