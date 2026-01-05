import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN FÍSICA ---
t = np.linspace(-2, 5, 1000) # Tiempo en milisegundos

def standard_ringdown(t):
    # Modelo de Ringdown clásico: Oscilación amortiguada (Q alto)
    # h(t) = A * exp(-t/tau) * cos(omega*t) para t > 0
    h = np.zeros_like(t)
    
    # Fase Inspiralling (t < 0)
    idx_insp = t < 0
    h[idx_insp] = 0.4 * np.cos(10 * t[idx_insp]) * np.exp(0.2 * t[idx_insp])
    
    # Fase Ringdown (t >= 0)
    idx_ring = t >= 0
    tau_ring = 1.5 # Decae lento
    omega_ring = 15
    h[idx_ring] = 0.4 * np.exp(-t[idx_ring] / tau_ring) * np.cos(omega_ring * t[idx_ring])
    
    return h

def magneto_viscous_silence(t):
    # Modelo MVR: Sobreamortiguado (Critically Damped / Overdamped)
    # Viscosidad saturada -> Q << 1
    h = np.zeros_like(t)
    
    # Fase Inspiralling (t < 0) - Igual que el estándar
    idx_insp = t < 0
    h[idx_insp] = 0.4 * np.cos(10 * t[idx_insp]) * np.exp(0.2 * t[idx_insp])
    
    # Fase Silencio (t >= 0)
    idx_silence = t >= 0
    # Decaimiento puramente exponencial, SIN oscilación (coseno)
    # Tau extremadamente corto (viscosidad máxima)
    tau_viscous = 0.15 
    h[idx_silence] = 0.4 * np.exp(-t[idx_silence] / tau_viscous) 
    
    return h

# --- GENERAR DATOS ---
h_standard = standard_ringdown(t)
h_mvr = magneto_viscous_silence(t)

# --- GRAFICAR ---
plt.figure(figsize=(8, 4))

# Señal Estándar (Fantasma de fondo)
plt.plot(t, h_standard, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Standard Ringdown (BH/NS)')

# Señal MVR (Tu teoría)
plt.plot(t, h_mvr, color='red', linewidth=3, label=r'Magneto-Viscous Silence ($Q \ll 1$)')

# Línea de Merger
plt.axvline(x=0, color='black', linestyle=':', alpha=0.3)

plt.xlabel('Time from Merger (ms)', fontsize=12)
plt.ylabel('GW Strain Amplitude h(t)', fontsize=12)
plt.title('Figure 2: The Sudden Silence', fontsize=14)
plt.legend()
plt.xlim(-2, 5)
plt.grid(alpha=0.3)

plt.savefig('fig2_silence_generated.png', dpi=300)
print("Figura 2 generada exitosamente.")
plt.show()