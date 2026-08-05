HIDDEN_DIM=256
TRAIN_EPISODES=300

echo "Starting Run 59: Burn-in | Gamma=0.0 | Layers=1 | M=30.0 | action_scale=5 | P.C. | lr=1e-4 | bs=128 | te=-2"
python run_job.py --mode 0 \
    --gamma 0.0 \
    --n_layers 1 \
    --update_period 1000 \
    --mismatch 30.0 \
    --hidden_dim $HIDDEN_DIM \
    --train_episodes $TRAIN_EPISODES \
    --n_shots 65000 \
    --burn_in_steps 15000 \
    --action_scale 5 \
    --lr 1e-4 \
    --batch_size 128 \
    --target_entropy -2.0 \
    --use_pearson_correlation \
    --model_path models/sac_gnn_59.pth \
    --training_metrics_filename training_metrics_59.png