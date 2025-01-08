import numpy as np
import matplotlib.pyplot as plt 

D = 100
Lx = 300

dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)

whos

x

x[0]

x[nx-1]

x[-1]

x[0:5]

C = np.zeros_like(x)
C_left = 500
C_right = 0
C[x <= Lx//2] = C_left
C[x > Lx//2] = C_right

C

plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial concentration profile")

time = 0 
nt = 5000
dt = 0.5 * (dx**2 / D)

dt

z = 5

z = z + 5
z

z += 5
z

for t in range(0, nt):
    C += D * dt / dx**2 * (np.roll(C, -1) - 2*C + np.roll(C, 1))
    C[0] = C_left
    C[-1] = C_right

plt.figure()
plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final concentration profile")

z = np.arange(5)
z

np.roll(z, -1)

np.roll(z, 1)

for i in range(len(z)):
    print(z[i] +1)

z + 1
