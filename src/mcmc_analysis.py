"""
MCMC Bayesian Analysis for Magnetar Spin-Down
Author: Orlando Miguel Urbina González (MVR Model Validation)
Description: 
  Metropolis-Hastings implementation to fit the Spin-Down model 
  L(t) = L0 * (1 + t/tau)^-2 to GRB 130603B data.
  It estimates the posterior distribution of the Magnetic Field (B).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. DATOS REALES (GRB 130603B) ---
# Tiempos (s) y Luminosidad (erg/s) extraídos de Rowlinson et al. (2013)
t_obs = np.array([300, 1000, 4000, 10000, 40000])
L_obs = np.array([0.9e47, 0.88e47, 0.7e47, 0.2e47, 0.02e47])
L_err = np.array([0.1e47, 0.1e47, 0.1e47, 0.05e47, 0.01e47]) # Barras de error

# Constantes Físicas
I = 2e45        # Inercia (g cm^2)
R = 1.2e6       # Radio (cm)
c = 3e10        # Velocidad luz (cm/s)
Omega0 = 3140.0 # Vel. angular inicial (rad/s) ~ 2ms

# --- 2. EL MODELO FÍSICO (Magnetar Spin-Down) ---
def get_tau(B_field):
    # Calcula tau_sd dado un campo B (en Gauss)
    # tau = 3 c^3 I / (B^2 R^6 Omega^2)
    numerator = 3 * (c**3) * I
    denominator = (B_field**2) * (R**6) * (Omega0**2)
    return numerator / denominator

def model_luminosity(t, B_field, L_initial):
    # L(t) = L0 * (1 + t/tau)^-2
    tau = get_tau(B_field)
    return L_initial * (1 + t/tau)**(-2)

# --- 3. ESTADÍSTICA BAYESIANA ---
def log_likelihood(theta, t, y, yerr):
    B_val, L_val = theta
    model = model_luminosity(t, B_val, L_val)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    B_val, L_val = theta
    # Priors planos (informados pero amplios)
    # B entre 10^14 y 10^16 G
    # L0 entre 10^46 y 10^48 erg/s
    if 1e14 < B_val < 1e16 and 0.5e47 < L_val < 1.5e47:
        return 0.0
    return -np.inf

def log_probability(theta, t, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y, yerr)

# --- 4. ALGORITMO MCMC (Metropolis-Hastings) ---
def run_mcmc(n_steps=10000):
    print("Iniciando MCMC... Buscando el mejor ajuste...")
    
    # Punto de partida (Guess inicial)
    current_theta = np.array([1e15, 0.9e47]) 
    
    chain = []
    accepted = 0
    
    for i in range(n_steps):
        # Propuesta de nuevo paso (salto aleatorio gaussiano)
        proposal = current_theta + np.random.normal(0, [0.5e14, 0.02e47])
        
        prob_current = log_probability(current_theta, t_obs, L_obs, L_err)
        prob_proposal = log_probability(proposal, t_obs, L_obs, L_err)
        
        # Criterio de aceptación
        if prob_proposal > prob_current:
            current_theta = proposal
            accepted += 1
        else:
            ratio = np.exp(prob_proposal - prob_current)
            if np.random.rand() < ratio:
                current_theta = proposal
                accepted += 1
        
        chain.append(current_theta)
    
    print(f"MCMC Terminado. Tasa de aceptación: {accepted/n_steps:.2%}")
    return np.array(chain)

# --- 5. EJECUCIÓN Y GRÁFICAS ---
if __name__ == "__main__":
    # Correr simulación
    chain = run_mcmc(n_steps=20000)
    
    # Quemar los primeros 20% pasos (Burn-in)
    burn_in = int(len(chain) * 0.2)
    samples = chain[burn_in:]
    
    B_samples = samples[:, 0]
    L_samples = samples[:, 1]
    
    # Resultados Estadísticos
    B_mean = np.mean(B_samples)
    B_std = np.std(B_samples)
    tau_best = get_tau(B_mean)
    
    print("-" * 40)
    print(f"RESULTADOS DEL AJUSTE BAYESIANO:")
    print(f"Campo Magnético (B): {B_mean/1e15:.3f} x 10^15 G  (+/- {B_std/1e15:.3f})")
    print(f"Tiempo de Spin-Down (tau): {tau_best:.0f} segundos")
    print("-" * 40)
    
    # --- GRÁFICA 1: AJUSTE CON BANDAS DE INCERTIDUMBRE ---
    plt.figure(figsize=(8, 6))
    
    # Graficar 100 modelos aleatorios de la cadena posterior
    t_fit = np.logspace(1.5, 5, 200)
    for i in np.random.randint(0, len(samples), 100):
        plt.plot(t_fit, model_luminosity(t_fit, samples[i,0], samples[i,1]), 
                 color='blue', alpha=0.05)
        
    # Mejor ajuste
    plt.plot(t_fit, model_luminosity(t_fit, B_mean, np.mean(L_samples)), 
             color='black', linewidth=2, label=f'Best Fit (tau={tau_best:.0f}s)')
    
    # Datos
    plt.errorbar(t_obs, L_obs, yerr=L_err, fmt='ro', capsize=4, label='GRB 130603B Data')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Luminosity (erg/s)')
    plt.title('Bayesian MCMC Fit: Magnetar Spin-Down')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('mcmc_fit_result.png')
    
    # --- GRÁFICA 2: DISTRIBUCIÓN POSTERIOR DE B (Evidencia) ---
    plt.figure(figsize=(6, 4))
    plt.hist(B_samples/1e15, bins=30, color='skyblue', edgecolor='black', density=True)
    plt.axvline(B_mean/1e15, color='red', linestyle='--', linewidth=2, label=f'Mean B = {B_mean/1e15:.2f}e15 G')
    
    plt.xlabel('External Magnetic Field ($10^{15}$ G)')
    plt.ylabel('Probability Density')
    plt.title('Posterior Distribution of Magnetic Field')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mcmc_B_distribution.png')
    
    print("Gráficas generadas: 'mcmc_fit_result.png' y 'mcmc_B_distribution.png'")