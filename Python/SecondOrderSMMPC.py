import casadi as cas
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
Wb = 2 * np.pi * 60
Pmax = 10
M = 15

# Definición del modelo de estado T(x, u)
def T(x, u):
    x1, x2 = x[0], x[1]  # Desempacar valores de x
    u1 = u[0]  # Desempacar valor de u
    
    # Calcular las derivadas
    dx1 = -Pmax/M * cas.sin(x2) + u1/M
    dx2 = Wb * (x1 - 1)
    
    return cas.vertcat(dx1, dx2)

# Parámetros de la simulación
nt = 200  # Número de pasos de tiempo
nx = 2  # Número de variables de estado
nu = 1  # Número de variables de control
dt = 1 / 60 / 4  # Paso de tiempo
alpha = 0.0001  # Peso para la penalización en la función objetivo

# Punto inicial
xini = np.array([0.8, 0.9])

# Modelo de optimización (Euler)
opti = cas.Opti()

# Horizonte de predicción
N = 10

# Definir variables de optimización
xk = opti.parameter(nx, 1)
xk1 = opti.variable(nx, N)
uopt = opti.variable(nu, N)

# Puntos de referencia
xref = np.array([1, np.pi/3])
uref = 10 / 15 * np.sin(np.pi/3)

# Función objetivo
L = (xk1[:,0] - xref).T @ (xk1[:,0] - xref) + alpha * (uopt[0] - uref)**2

for k in range(1, N):
    L += (xk1[:,k] - xref).T @ (xk1[:,k] - xref) + alpha * (uopt[k] - uref)**2
    opti.subject_to(xk1[:,k] == xk1[:,k-1] + dt * T(xk1[:,k-1], uopt[:,k-1]))

opti.subject_to(xk1[:,0] == xk)
opti.minimize(L)

# Resolver el problema de optimización
opti.solver('ipopt')

# Parámetros de simulación
tode = np.linspace(0, (nt-1)*dt, nt)
xode = np.zeros((nx, nt))
uode = np.zeros((nu, nt))

# Punto inicial
x = xini

# Función de control
def ucontrol(x, opti, xk, uopt):
    opti.set_value(xk, x)
    sol = opti.solve()
    u = sol.value(uopt[0])
    return u

# Guardar el primer paso
xode[:,0] = x
uode[:,0] = ucontrol(x, opti, xk, uopt)

# Simulación utilizando Runge-Kutta de cuarto orden
# Simulación utilizando Runge-Kutta de cuarto orden
for k in range(1, nt):
    f1 = T(x, uode[:, k-1])
    f2 = T(x + dt*f1/2, uode[:, k-1])
    f3 = T(x + dt*f2/2, uode[:, k-1])
    f4 = T(x + dt*f3, uode[:, k-1])

    x = x + dt*(f1 + 2*f2 + 2*f3 + f4) / 6
    u = ucontrol(x, opti, xk, uopt)
    #u = np.pi/3
    # Asegurar que x sea un vector unidimensional
    x = np.squeeze(x)

    xode[:, k] = x
    uode[:, k] = u

# Graficar los resultados de la simulación
plt.figure()

plt.subplot(3,1,1)
plt.plot(tode, xode[0,:])
plt.ylabel('\omega')
plt.xlabel('t')
plt.title('Simulación del sistema: Velocidad angular (\omega)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(tode, xode[1,:])
plt.ylabel('\delta')
plt.xlabel('t')
plt.title('Simulación del sistema: Ángulo (\delta)')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(tode, uode[0,:])
plt.ylabel('P_M')
plt.xlabel('t')
plt.title('Simulación del sistema: Potencia Mecánica (P_M)')
plt.grid(True)

plt.tight_layout()
plt.show()
