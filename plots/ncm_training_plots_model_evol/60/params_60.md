HIDDEN_DIM=256
TRAIN_EPISODES=300

echo "Starting Run 60: Burn-in | Gamma=0.0 | Layers=1 | M=30.0 | action_scale=5 | Using P.C. | lr=1e-4 | bs=256 | te=-0.5"
python run_job.py --mode 0 \
    --gamma 0.0 \
    --n_layers 1 \
    --update_period 1000 \
    --mismatch 30.0 \
    --hidden_dim $HIDDEN_DIM \
    --n_shots 65000 \
    --burn_in_steps 15000 \
    --action_scale 5 \
    --lr 1e-4 \
    --batch_size 256 \
    --target_entropy -0.5 \
    --use_pearson_correlation \
    --model_path models/sac_gnn_60.pth \
    --training_metrics_filename training_metrics_60.png