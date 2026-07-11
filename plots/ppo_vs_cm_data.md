--model-path models/linear/linear_model_ppo_0_best.pth  
Linear-CM agent vs analytical CM | distance=5, mismatch=1.0
Model: models/linear/linear_model_ppo_0_best.pth
p values: ['2.00e-04', '3.09e-04', '4.77e-04', '7.37e-04', '1.14e-03', '1.76e-03', '2.71e-03', '4.19e-03', '6.47e-03', '1.00e-02']
Target errors per p: 500 | Max shots per p: 10,000,000,000,000

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=2.000e-04 | shots=712,000,000 ---
  MWPM   :    813 err | LER 1.142e-06 +/- 4.0e-08
  CM     :    659 err | LER 9.256e-07 +/- 3.6e-08
  Neural :    500 err | LER 7.022e-07 +/- 3.1e-08
  -> Neural/CM LER ratio: 0.759 (Neural better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=3.089e-04 | shots=212,000,000 ---
  MWPM   :    793 err | LER 3.741e-06 +/- 1.3e-07
  CM     :    619 err | LER 2.920e-06 +/- 1.2e-07
  Neural :    504 err | LER 2.377e-06 +/- 1.1e-07
  -> Neural/CM LER ratio: 0.814 (Neural better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=4.771e-04 | shots=57,000,000 ---
  MWPM   :    808 err | LER 1.418e-05 +/- 5.0e-07
  CM     :    533 err | LER 9.351e-06 +/- 4.1e-07
  Neural :    507 err | LER 8.895e-06 +/- 4.0e-07
  -> Neural/CM LER ratio: 0.951 (Neural better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=7.368e-04 | shots=15,000,000 ---
  MWPM   :    840 err | LER 5.600e-05 +/- 1.9e-06
  CM     :    518 err | LER 3.453e-05 +/- 1.5e-06
  Neural :    517 err | LER 3.447e-05 +/- 1.5e-06
  -> Neural/CM LER ratio: 0.998 (Neural better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=1.138e-03 | shots=5,000,000 ---
  MWPM   :    948 err | LER 1.896e-04 +/- 6.2e-06
  CM     :    558 err | LER 1.116e-04 +/- 4.7e-06
  Neural :    593 err | LER 1.186e-04 +/- 4.9e-06
  -> Neural/CM LER ratio: 1.063 (CM better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=1.758e-03 | shots=2,000,000 ---
  MWPM   :   1419 err | LER 7.095e-04 +/- 1.9e-05
  CM     :    875 err | LER 4.375e-04 +/- 1.5e-05
  Neural :    921 err | LER 4.605e-04 +/- 1.5e-05
  -> Neural/CM LER ratio: 1.053 (CM better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=2.714e-03 | shots=1,000,000 ---
  MWPM   :   2563 err | LER 2.563e-03 +/- 5.1e-05
  CM     :   1697 err | LER 1.697e-03 +/- 4.1e-05
  Neural :   1822 err | LER 1.822e-03 +/- 4.3e-05
  -> Neural/CM LER ratio: 1.074 (CM better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=4.192e-03 | shots=1,000,000 ---
  MWPM   :   8305 err | LER 8.305e-03 +/- 9.1e-05
  CM     :   5714 err | LER 5.714e-03 +/- 7.5e-05
  Neural :   6073 err | LER 6.073e-03 +/- 7.8e-05
  -> Neural/CM LER ratio: 1.063 (CM better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=6.475e-03 | shots=1,000,000 ---
  MWPM   :  27624 err | LER 2.762e-02 +/- 1.6e-04
  CM     :  21252 err | LER 2.125e-02 +/- 1.4e-04
  Neural :  21939 err | LER 2.194e-02 +/- 1.5e-04
  -> Neural/CM LER ratio: 1.032 (CM better)

[*] Models successfully loaded from models/linear/linear_model_ppo_0_best.pth
--- p=1.000e-02 | shots=1,000,000 ---
  MWPM   :  81746 err | LER 8.175e-02 +/- 2.7e-04
  CM     :  69615 err | LER 6.961e-02 +/- 2.5e-04
  Neural :  71011 err | LER 7.101e-02 +/- 2.6e-04
  -> Neural/CM LER ratio: 1.020 (CM better)


Saved plot to plots\ler_vs_p_linear_ppo_vs_cm.png
Saved raw results to plots\ler_vs_p_linear_ppo_vs_cm.npz