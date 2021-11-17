

import time
import cvxpy as cp
from cvxpy.expressions.cvxtypes import constant
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import model

CELCIUS = 273.15

def exit():
    import sys
    sys.exit(0)

def get_model(t_delta):

    ha, k = [184.41449948998206, 23505513.644202553]

    A = np.array([-ha/k])
    B = np.array([1/k, 1/k, ha/k])
    #B = np.array([1/k])
    #F = np.array([1/k, ha/k])

    C = np.array([1])
    D = np.array([0, 0, 0])
    #D = np.array([0])

    css = signal.StateSpace(A, B, C, D)
    dss = css.to_discrete(t_delta)

    return dss.A, dss.B

def simulate(n_steps, t_delta, u, x_0, t_out, electric):
    A, B = get_model(t_delta)
    print(t_delta)

    x = np.zeros((n_steps + 1,))
    x[0] = x_0[0]
    for n in range(n_steps):
        #_t = t[n]
        Q_gas = 0
        
        inputs = np.array([ Q_gas, electric, t_out[n] ])

        #print(n, B, "______", inputs)
        #x[n + 1] = A * x[n] + B @ inputs
        x[n + 1] = A * x[n] + B @ inputs

    return x[:-1]



def simulate3(n_steps, t_delta, u, x_0):

    #Simulation Parameters
    x0 = [0,0]
    start = 0
    stop = 30
    step = 1
    t = np.arange(start,stop,step)
    K = 3
    T = 4
    # State-space Model
    A = [[-1/T, 0],
    [0, 0]]
    B = [[K/T],
    [0]]
    C = [[1, 0]]
    D = 0
    sys = signal.StateSpace(A, B, C, D)
    # Step Response
    t, y = signal.step(sys, x0, t)
    # Plotting
    plt.plot(t, y)
    plt.title("Step Response")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid()
    plt.show()

    exit()

def simulate2(n_steps, t_delta, u, x_0):

    ha, k = [184.41449948998206, 23505513.644202553]

    A = np.array([-ha/k])
    B = np.array([1/k, 1/k, ha/k])

    C = np.array([1])
    D = np.array([0, 0, 0])

    css = signal.StateSpace(A, B, C, D)

    x = np.matrix([[1]])
    u = np.matrix([[1],[1],[1]])

    print(A @ x )
    print(B @ u )
    print(C @ x )
    print(D @ u )
    print(A @ x + B @ u )
    print(C @ x + D @ u )

    print(css)

    d = np.array([1900, CELCIUS + 12.0])

    t = np.arange(0,10*60*60, 100)
    tu = np.array([t, t, t])

    # Step Response
    t, y = signal.step(css, x_0, tu)
    # Plotting
    plt.plot(t, y)
    plt.title("Step Response")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid()
    plt.show()

    exit()

if __name__ == "__main__":

    sim_time = 18*60*60
    n_steps = 300
    t_delta = sim_time / 200

    #t_delta = 10*60
    #n_steps = int(50*60*60/t_delta)

    x_0 = [22 + CELCIUS]

    t = np.array(range(n_steps)) * t_delta

    ref = np.zeros_like(t)
    for n in range(len(ref)):
        if n < n_steps/3:
            ref[n] = 19
        elif n < n_steps*2/3:
            ref[n] = 22
        else:
            ref[n] = 25
    ref = ref + CELCIUS

    x = cp.Variable(n_steps + 1)
    u = cp.Variable(n_steps)

    d = np.array([1000, CELCIUS + 12.0])

    A, B = get_model(t_delta)

    constraints = []
    costs = []
    for n in range(n_steps):
        inputs = [u[n], d[0], d[1]]
        constraints += [ x[n + 1] == A * x[n] + B[0] @ inputs ]
        constraints += [ u[n] <= 8000 ]
        constraints += [ u[n] >= 0 ]
        #print(ref[n])
        #print(x[n])
        costs += [ cp.sum_squares(ref[n] - x[n]) ]
        costs += [ 1000*cp.pos(ref[n] - x[n]) ]
    
    constraints += [ x[0] == x_0 ]
    #costs += cp.sum_squares(u)
    #costs += cp.sum_squares(x - ref)
    objective = cp.Minimize(cp.sum(costs))

    #objective = cp.Minimize(cp.sum_squares(x[:-1] - ref))

    prob = cp.Problem(objective, constraints)

    # cp.GLPK_MI
    # cp.ECOS

    print("Optimal value", prob.solve(verbose=True, solver=cp.ECOS))
    print("Optimal x")
    print(x.value) # A numpy ndarray.
    print("Optimal u")
    print(u.value) # A numpy ndarray.


    t = np.array(range(n_steps)) * t_delta
    x_sim = simulate(n_steps, t_delta, np.zeros((n_steps,)), x_0, np.ones_like(t)*d[1], d[0])
    ha, k = [184.41449948998206, 23505513.644202553]
    x_sim_cont = model.solve(x_0[0], t, np.ones_like(t)*d[1], np.zeros_like(t), np.ones_like(t)*d[0]+20, k, ha)


    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(t, ref, label='setpoint', linestyle="dashed", color='black')
    ax1.plot(t, x.value[:-1], label='temperature', color='blue')
    #ax1.plot(t, x_sim, label='x_sim', color='green')
    #ax1.plot(t, x_sim_cont, label='x_sim cont', color='red')
    ax2.plot(t, u.value, label='u (gasspeis)', color='orange')
    
    ax1.set_xlabel('t')
    ax1.set_ylabel('Tempearture [C]')
    ax2.set_ylabel('Gasspeis [W]')
    plt.title('Temperature MPC')
    ax1.yaxis.set_major_formatter(lambda x, pos: "{:.2f}".format(x-CELCIUS))
    ax1.xaxis.set_major_formatter(lambda x, pos: "{:.0f}h".format(x/3600.0))
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    #ax1.set_ylim([CELCIUS, CELCIUS + 40])
    plt.grid()
    plt.show()