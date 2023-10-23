HMM.py:

	Instruction to run script:
		- dependencies: scipy, pandas, numpy
		- data file "sequences.mat" in same directory

	Code description:
		- prints number of iterations, followed by the sequences in each cluster
		- 4 main functions: 
			1) calculate_tau() -> returns matrix of tau_k_n
			2) update_pi() -> returns updated parameters for choosing each cluster
			3) update_eta() -> returns updated parameters for initial distribution in each Markov Chain
			4) update_zeta() -> returns updated parameters for transition probabilities in each Markov Chain