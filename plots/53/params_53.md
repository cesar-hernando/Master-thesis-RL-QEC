HIDDEN_DIM=256
TRAIN_EPISODES=500

echo "Starting Run 53: 500 eps | No burn-in | Gamma=0.99 | Layers=1 | Mismatch=30.0 | action_scale=5 | Using P.C."
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
    --model_path models/sac_gnn_53.pth \
    --training_metrics_filename training_metrics_53.png