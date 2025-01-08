import numpy as np
import matplotlib.pyplot as plt
import control
import math

lr = 1.39
lf = 1.55
Ca = 20000
Iz = 25854
m = 1888.6
g = 9.81

vs = [i for i in range(1,41)]
ratios = []
p1 = []
p2 = []
p3 = []
p4 = []

for v in vs:
    A = np.array([[0, 1, 0, 0], \
                  [0, -4*Ca/(m*v), 4*Ca/m, -2*Ca*(lf-lr)/(m*v)], \
                  [0, 0, 0, 1], \
                  [0, -2*Ca*(lf-lr)/(Iz*v), 2*Ca*(lf-lr)/Iz, -2*Ca*(lf**2+lr**2)/(Iz*v)]])
    B = np.array([[0, 0], \
                  [2*Ca/m, 0], \
                  [0, 0], \
                  [2*Ca*lf/Iz, 0]])
    C = np.array([1, 1, 1, 1])
    
    P = control.ctrb(A, B)
    sv = np.linalg.svdvals(P)
    ratios.append(math.log10(sv[0] / sv[-1]))

    sys = control.StateSpace(A, B, C, 0)
    poles = control.poles(sys)
    p1.append(poles[0].real)
    p2.append(poles[1].real)
    p3.append(poles[2].real)
    p4.append(poles[3].real)

plt.figure(dpi=100)
plt.plot(vs, ratios)
plt.xlabel("v (m/s)")
plt.ylabel("log_10(s_1 / s_n)")

fig2, axs = plt.subplots(4, 1, sharex='all')
axs[0].plot(vs, p1)
axs[0].set_ylabel("p1")
axs[1].plot(vs, p2)
axs[1].set_ylabel("p2")
axs[2].plot(vs, p3)
axs[2].set_ylabel("p3")
axs[3].plot(vs, p4)
axs[3].set_ylabel("p4")

plt.xlabel("v (m/s)")

plt.show()

print("The higher the velocity gets, the closer the system gets to being unstable and uncontrollable")