import numpy as np
from scipy.integrate import odeint
from scipy import signal


class Model:

    # Best ha, k found: [184.41449948998206, 23505513.644202553]
    def __init__(self):
        self.ha = 184.41449948998206
        self.k = 23505513.644202553

        self.update_model()
        

    def set_ha(self, ha):
        self.ha = ha

    def set_k(self, k):
        self.k = k

    def update_model(self):

        # The actual ODE is: T_dot = ( ha*(T - T_env) + Q_electic + Q_gas ) / k
        # 
        # Where:
        #  - T: temperature inside [K]
        #  - T_env: temperature outside [K]
        #  - Q_electric: the electric power injected into the system (input disturbance) [W] 
        #  - Q_gas: power from the gas fireplace (controlled input) [W]
        #  - ha: combination of h (heat transfer coefficient) and A (surface area) [WK]
        #  - k: heat capacity of the system 
        # 
        # The LTI is build around:
        #  - x: [T]
        #  - u: [Q_gas, Q_electric, T_env] (where Q_gas is a controlled input, the others are disturbances)

        A = np.array([-self.ha/self.k])
        B = np.array([1/self.k, 1/self.k, self.ha/self.k])

        C = np.array([1])
        D = np.array([0, 0, 0])

        self.A = A
        self.B = B
        self.C = C
        self.D = D

    # Get the continuous state space model
    def get_model_continuous(self):
        return self.A, self.B, self.C, self.D

    # Get the discrete state space model
    def get_model_discrete(self, dt):

        css = signal.StateSpace(self.A, self.B, self.C, self.D)
        dss = css.to_discrete(dt)

        return dss.A, dss.B, dss.C, dss.D

    # Evaluate the ODE (calcuate T_dot)
    def eval_ode(self, y, t, u, d):

        # u: Q_gas (controlled input) with timestamps, example
        # u = (t, Q_gas) "Q_gas" is a list of values, at the times given in t

        # d: input disturbances with timestamps, example
        # [ (t, Q_electric),
        #   (t, T_env) ]
        #
        # "Q_electric" is a list of values, at the times given in t

        electric = d[0]
        t_env = d[1]
        gas = u

        Q_electric =  np.interp(t, electric[0], electric[1]) #1900 #1200.0
        Q_gas = np.interp(t, gas[0], gas[1])
        t_out_v = np.interp(t, t_env[0], t_env[1])
    
        # TODO switch to using state space model instead of the ode directly
        return (Q_electric + Q_gas - self.ha*(y[0] - t_out_v)) / self.k

    # Solve the ODE for given t, u, d
    def solve_continuous(self, y0, t, t_environment, gas, electric):
        sol = odeint(self.eval_ode, [y0], t, args=((t, gas), [(t, electric), (t, t_environment)])).flatten()

        return sol

    def solve_discrete(self, y0, t, t_environment, gas, electric):
    
        Ak, Bk, Ck, Dk = self.get_model_discrete(t[1] - t[0])

        x = np.zeros((t.shape[0] + 1,))
        x[0] = y0
        for n in range(t.shape[0]):
            _t = t[n]
            Q_electric = np.interp(_t, t, electric) #1900 #1200.0
            Q_gas = np.interp(_t, t, gas)
            t_out_v = np.interp(_t, t, t_environment)

            inputs = np.array([ Q_gas, Q_electric, t_out_v ])

            #print(n, B, "______", inputs)
            #x[n + 1] = A * x[n] + B @ inputs
            x[n + 1] = Ak * x[n] + Bk @ inputs

        return x[:-1]

# TODO replaced by Model.eval_ode()
def model(y, t, ha, t_out, k, gas, electric):

    #ha = 0.01
    #t_out = 2

    #k = 11000000

    Q_electric = np.interp(t, electric[0], electric[1]) #1900 #1200.0
    Q_gas = np.interp(t, gas[0], gas[1])
    t_out_v = np.interp(t, t_out[0], t_out[1])
    
    return (Q_electric + Q_gas - ha*(y[0] - t_out_v)) / k

# TODO replace with Model.xxx
def solve(y0, t, t_out, gas, electric, k, ha):

    sol = odeint(model, [y0], t, args=(ha, (t, t_out), k, (t, gas), (t, electric)))

    return sol

# TODO replace with Model.solve_continuous
def solve_disc(y0, t, t_out, gas, electric, k, ha):

    A = np.array([-ha/k])
    B = np.array([1/k, 1/k, ha/k])
    #B = np.array([1/k])
    #F = np.array([1/k, ha/k])

    C = np.array([1])
    D = np.array([0, 0, 0])
    #D = np.array([0])

    css = signal.StateSpace(A, B, C, D)
    dss = css.to_discrete(t[1] - t[0])
    print(css)
    print(dss.A)

    Ak = dss.A
    Bk = dss.B

    x = np.zeros((t.shape[0] + 1,))
    x[0] = y0
    for n in range(t.shape[0]):
        _t = t[n]
        Q_electric = np.interp(_t, t, electric) #1900 #1200.0
        Q_gas = np.interp(_t, t, gas)
        t_out_v = np.interp(_t, t, t_out)

        inputs = np.array([ Q_gas, Q_electric, t_out_v ])

        #print(n, B, "______", inputs)
        #x[n + 1] = A * x[n] + B @ inputs
        x[n + 1] = Ak * x[n] + Bk @ inputs

    return x[:-1]


if __name__ == "__main__":

    y0 = [22.0+273]
    t = np.linspace(0, 3*24*60*60, 101)

    gas_array = []
    for tm in t:
        if tm < 20000:
            gas_array.append(0)
        elif tm < 60000:
            gas_array.append(2850)
        else:
            gas_array.append(0)
    gas = np.array(gas_array)

    has = np.linspace(500, 5000, 10)
    has = [190]

    ks = np.linspace(110000, 11000000, 10)
    #ks = [2600000]
    ks = [23505513]

    t_out = np.ones_like(t) * 2+273


    electric = np.ones_like(t) * 1200

    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for ha in has:
        for k in ks:

            ss_model = Model()

            sol = odeint(model, y0, t, args=(ha, (t, t_out), k, (t, gas), (t, electric)))
            sol_2 = odeint(ss_model.eval_ode, y0, t, args=((t, gas), [(t, electric), (t, t_out)]))
            ax1.plot(t, sol[:, 0], label='ha: '+str(ha)+ " k:"+str(k))
            ax1.plot(t, sol_2[:, 0], label='2 ha: '+str(ha)+ " k:"+str(k))

    plt.xlabel('t')
    ax1.xaxis.set_major_formatter(lambda x, pos: "{:.0f}h".format(x/3600.0))
    ax1.yaxis.set_major_formatter(lambda x, pos: "{:.0f}".format(x-273))
    ax1.legend(loc='upper left')
    ax2.plot(t, gas, label='gas', color='orange')
    ax2.legend(loc='upper right')
    plt.grid()
    plt.show()

