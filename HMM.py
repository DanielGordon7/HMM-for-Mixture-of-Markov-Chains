import scipy.io as sio
import pandas as pd
import numpy as np
from numpy.random import default_rng

# read sequence data and stored in list
sequences_data = sio.loadmat("sequences.mat")["sequences"][0]
seq_vec = [seq[0] for seq in sequences_data]

def calculate_tau(seq_vec, labels, pi, eta, zeta, K, N):
    tau = np.zeros((N, K))
    for n in range(1, N+1):
        cur_seq = seq_vec[n-1]
        for k in range(1, K+1):
            # probability of selecting each Markov Chain
            rng = default_rng()
            h_sample = rng.multinomial(1, pi[k-1])
            tao_k_n = pi[k-1][np.where(h_sample == 1)[0][0]]

            # initial distribution of Markov Chain
            first_letter_num = labels[cur_seq[0]]
            tao_k_n *= eta[k-1][first_letter_num-1]  

            # transition probabilities of Markov Chain
            for t in range(1, T):
                t_letter_num = labels[cur_seq[t-1]]
                t_plus1_letter_num = labels[cur_seq[t]]
                tao_k_n *= zeta[k-1][t_letter_num-1][t_plus1_letter_num-1]
            tau[n-1, k-1] = tao_k_n
    # normalize
    tau /= np.sum(tau, axis=1).reshape(20, 1)
    return tau

def update_pi(tau, K, N):
    # parameter for the probability of selecting each cluster (Markov Chain)
    new_pi = np.zeros((K, K))
    for k in range(1, K+1):
        for n in range(1, N+1):
            new_pi[k-1] += tau[n-1][k-1]
        new_pi[k-1] /= N
    return new_pi

def update_eta(tau, labels, K, N, V):
    # parameter for probability of sampling each DNA letter in first observation
    new_eta = np.zeros((K, V))
    for k in range(1, K+1):
        for letter, r in labels.items():
            for n in range(1, N+1):
                if (seq_vec[n-1][0] == letter):
                    new_eta[k-1][r-1] += tau[n-1][k-1]
            new_eta[k-1][r-1] /= np.sum(tau[:, k-1], axis=0)
    return new_eta

def update_zeta(tao, labels, K, N, V, T):
    # parameter for transition probability between DNA letters in consecutive observations
    new_zeta = np.zeros((K, V, V))
    for letter_t, r_t in labels.items():
        for letter_tplus1, r_tplus1 in labels.items():
            for k in range(1, K+1):
                numerator = denominator = 0
                for n in range(1, N+1):
                    cur_letter = seq_vec[n-1][0]
                    for t in range(1, T):
                        next_letter = seq_vec[n-1][t]

                        if cur_letter == letter_t:
                            denominator += tao[n-1][k-1]
                            if next_letter == letter_tplus1:
                                numerator += tao[n-1][k-1]
                        cur_letter = next_letter

                    if (cur_letter == letter_t):
                        denominator += tao[n-1][k-1]

                new_zeta[k-1][r_t-1][r_tplus1-1] = numerator / denominator
    return new_zeta


# data
N = T = 20
V = 4
K = 2
labels = {"A": 1, "C": 2, "G": 3, "T": 4}

# initialize parameters randomly
rng = default_rng()
pi_uniform = rng.uniform(0, 1, (K, K))
pi = pi_uniform / np.sum(pi_uniform, axis=1).reshape((K, 1))
eta_uniform = rng.uniform(0, 1, (K, V))
eta = eta_uniform / np.sum(eta_uniform, axis=1).reshape((K, 1))
zeta_uniform = rng.uniform(0, 1, (K, V, V))
zeta = zeta_uniform / np.sum(zeta_uniform, axis=2).reshape((K, V, 1))

# EM algorithm
max_itr = 100
cur_itr = 0
while (cur_itr < max_itr):
    tau = calculate_tau(seq_vec, labels, pi, eta, zeta, K, N)
    pi = update_pi(tau, K, N)
    eta = update_eta(tau, labels, K, N, V)
    zeta = update_zeta(tau, labels, K, N, V, T)
    new_tau = calculate_tau(seq_vec, labels, pi, eta, zeta, K, N)
    # exit while loop if converged
    if np.isclose(tau, new_tau).sum() == 40:
        break
    cur_itr += 1

print(f"Number of iterations: {cur_itr}")

# classify sequences into clusters based on highest tau value
cluster1 = []
cluster2 = []
for n in range(1, N+1):
    if tau[n-1][0] > tau[n-1][1]:
        cluster1.append(seq_vec[n-1])
    else:
        cluster2.append(seq_vec[n-1])

print("\n Cluster 1:")
for seq in cluster1:
    print(seq)

print("\n Cluster 2:")
for seq in cluster2:
    print(seq)
