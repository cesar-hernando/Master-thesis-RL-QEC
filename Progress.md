# Progress Tracker

- (8/1/2026): Developed a measurement error-free rotated surface code gym environment and a DQN implementation (StableBaselines3) with CNN architecture that learns to decode errors. Currently, it learns to decode for distance 3, 5 and (with lower success rate) 7 (3e6 training steps for 5 and 7).

Questions: 
- Provided the remarkable results of AlphaQubit2 (and previous versions) that formulate the decoding task as a prediction of logical observables (supervised learnining classification), and afterwards update the Pauli frame (they do not explicitly do that as they just perform Z memory experiments), how can RL decoding, which outputs data qubits corrections (scale as d^2), bring value?

  Answer: A RL data qubit correction agent can be able to decode errors in the formal sense (logical frame), since data qubit corrections can be translated to Pauli frame updates. However, the number of possible corrections (action space) scale as d^2, as well as the expected number of corrections that are necessary (for a fixed noise).

- A key aspect of quantum error decoding, which AlphaQubit2 still struggles with (although they have made significant progress) is scalability (in terms of distance). Is it possible to design a multi-agent RL framework in which each agent decodes a local patch of the code, and then, for example, a "master" agent combines the action (or something else?) and solves the discrepancies? Can this improve scalability compared to a single monolithic agent? Can distance d trained agents be used to train faster a distance d+2 code?

  Answer: Using smaller patches to decode effectively reduces the distance of the code, unless there is a sufficiently large buffer zone. This could be interesting to explore.

- (16/02/2026): After performing a literature review of quantum error decoding, specialized in the surface code, I decided to change the approach from implementing a RL decoder that outputs corrections to data qubits, to a RL reweighter of MWPM that can adapt to time-varying noise and complex correlations. This approach is inspired by the paper: "DGR: Tackling Drifted and Correlated Noise in Quantum Error Correction via Decoding Graph Re-weighting". As a first step of this work, I simulate a noisy syndrome extraction circuit for the surface code in Stim. I also analyze how to generate a decoding graph, visualize it, sample syndrome measurements, and modify edge weights.

