HIDDEN_DIM=256
TRAIN_EPISODES=500

echo "Starting Run 63: Burn-in | Gamma=0.0 | Layers=1 | M=30.0 | action_scale=5 | Using P.C. | lr=5e-5 | bs=128 | te=-1"
python run_job.py --mode 0 \
    --gamma 0.0 \
    --n_layers 1 \
    --update_period 1000 \
    --mismatch 30.0 \
    --hidden_dim $HIDDEN_DIM \
    --n_shots 65000 \
    --burn_in_steps 15000 \
    --train_episodes $TRAIN_EPISODES \
    --action_scale 5 \
    --lr 5e-5 \
    --batch_size 128 \
    --target_entropy -1.0 \
    --use_pearson_correlation \
    --model_path models/sac_gnn_63.pth \
    --training_metrics_filename training_metrics_63.png