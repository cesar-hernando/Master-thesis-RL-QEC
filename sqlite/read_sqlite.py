import optuna

# Replace with your actual study name and db filename!
study_name = "my_qec_study" 
storage_url = "sqlite:///your_database_name.db"

study = optuna.load_study(study_name=study_name, storage=storage_url)

print(f"Total trials finished: {len(study.trials)}")
print(f"Best trial ID: {study.best_trial.number}")
print(f"Best LER (Value): {study.best_trial.value}")
print("Best Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")