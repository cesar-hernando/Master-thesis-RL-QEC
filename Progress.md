# Progress Tracker

- (8/1/2026): Developed a measurement error-free rotated surface code gym environment and a DQN implementation (StableBaselines3) with CNN architecture that learns to decode errors. Currently, it learns to decode for distance 3, 5 and (with lower success rate) 7 (3e6 training steps for 5 and 7).

Questions: 
- Provided the remarkable results of AlphaQubit2 (and previous versions) that formulate the decoding task as a prediction of logical observables (supervised learnining classification), and afterwards update the Pauli frame (they do not explicitly do that as they just perform Z memory experiments), how can RL decoding, which outputs data qubits corrections (scale as d^2), bring value?

- A key aspect of quantum error decoding, which AlphaQubit2 still struggles with (although they have made significant progress) is scalability (in terms of distance). Is it possible to design a multi-agent RL framework in which each agent decodes a local patch of the code, and then, for example, a "master" agent combines the action (or something else?) and solves the discrepancies? Can this improve scalability compared to a single monolithic agent? Can distance d trained agents be used to train faster a distance d+2 code?

