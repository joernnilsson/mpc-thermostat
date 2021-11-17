from homeassistant_api import Client, Entity, State
from datetime import datetime, time, timedelta, timezone
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize, least_squares

import model
import os


CELCIUS = 273.15


def fill_history(time_series, datetime_start, raw_json, transform = lambda x: x, method = 'interpolate'):
    data_time = []
    data_value = []
    for h in raw_json:
        state = h["state"]
        if state == "unavailable":
            continue
        data_value.append(transform(state))
        data_time.append((datetime.fromisoformat(h["last_updated"]).replace(tzinfo=None) - datetime_start).total_seconds())
    
    if method == 'interpolate':
        return np.interp(time_series, data_time, data_value)
    elif method == 'last':
        last_val = data_value[0]
        last_idx = 0
        out = []
        for t in time_series:
            next_idx = last_idx + 1
            if next_idx < len(data_time) and data_time[next_idx] < t:
                last_idx = next_idx
                last_val = data_value[last_idx]
            out.append(last_val)
        return np.array(out)


def fetch_ha_data():

    HA_TOKEN = os.environ['HA_TOKEN']

    client = Client(
        'https://ha.sf.while1.no/api',
        HA_TOKEN
    )

    date_end = datetime.now()
    date_start = date_end - timedelta(days = 1)

    ent_temp_stue = client.get_entity(entity_id='sensor.lumi_lumi_sens_temperature')
    ent_temp_ute = client.get_entity(entity_id="sensor.lumi_lumi_weather_5bd0c802_temperature")
    ent_fireplace = client.get_entity(entity_id="switch.pg_fireplace_switch")
    ent_energy = client.get_entity(entity_id="sensor.ams_active_power_import_6970631404129954")
    
    hist = client.get_history([ent_temp_stue, ent_temp_ute, ent_fireplace, ent_energy], timestamp=date_start, end_timestamp=date_end)

    time_series = np.linspace(0, (date_end - date_start).total_seconds() - 5*60*60, 301)

    data_temp_stue = fill_history(time_series, date_start, hist[0], lambda x: float(x) - CELCIUS)
    data_temp_ute = fill_history(time_series, date_start, hist[1], lambda x: float(x) - CELCIUS)
    data_fireplace = fill_history(time_series, date_start, hist[2], lambda x: 1 if x == 'on' else 0, method='last')
    data_energy = fill_history(time_series, date_start, hist[3], float)


    np.save("1_days.dat", np.array([time_series, data_temp_stue, data_temp_ute, data_fireplace, data_energy]))

if __name__ == "__main__":



    #k = 2600000
    #ha = 190
    fireplace_power = 8612 # Watts

    # 1 day
    #best_match = [109.31611509385789, 4487540.645292436, 2844.2534891671194] # 1_day
    best_match = [214.46979864892342, 26228265.668285176]

    # 3 day
    #best_match = [184.35155521392866, 23428086.025723375, 8612.694478443613] # 3_day
    best_match = [184.41449948998206, 23505513.644202553]

    ha = best_match[0]
    k = best_match[1]
    #fireplace_power = best_match[2]

    #data = np.load("1_days.dat.npy")
    data = np.load("3_days.dat.npy")

    time_series = data[0,:].flatten()
    data_temp_stue = data[1,:].flatten()
    data_temp_ute = data[2,:].flatten()
    data_fireplace = data[3,:].flatten()
    data_energy = data[4,:].flatten()

    data_temp_ute = data_temp_ute - 3 # data is collected in the shed and is offset by a few degrees
    

    # Search for best constants
    def cost(x):
        _ha, _k = x
        _f = fireplace_power

        ss_model = model.Model()
        ss_model.set_ha(_ha)
        ss_model.set_k(_k)
        ss_model.update_model()

        _est_temp_stue = ss_model.solve_continuous(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*_f, data_energy)
        #_est_temp_stue = model.solve(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*_f, data_energy, _k, _ha).flatten()

        diff = _est_temp_stue - data_temp_stue

        #print(diff)
        #import sys
        #sys.exit(0)
        
        cst = np.sum(np.square(diff))
        print(np.array([_ha, _k, _f]).tolist(), cst)

        return cst

    def ls_cost(x):
        ha, k, fireplace_power = x #np.multiply(x, x_scale)
        est_temp_stue = model.solve(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fireplace_power, data_energy, k, ha).flatten()

        diff = est_temp_stue - data_temp_stue

        #print(est_temp_stue.shape, data_temp_stue.shape, diff.shape)

        return diff


    x0 = np.array([ha, k]) #, fireplace_power]) # ha k

    est_temp_stue_m = None
    if True:

        # Search for the k and ha that makes the model match reality

        result = minimize(cost, x0, method = 'Nelder-Mead', args = (), bounds=[(0, np.Inf), (0, np.Inf)], options={'xatol': 0.01, 'fatol': 0.01})
        #result = least_squares(ls_cost, x0)

        print(result)
        x_sol = result.x

        #k = result.x[0]
        #ha = result.x[1]
        #fireplace_power = result.x[2]
        print("Solution: ", x_sol.tolist())
        print(data_fireplace)

        ss_model = model.Model()
        ss_model.set_ha(x_sol[0])
        ss_model.set_k(x_sol[1])
        ss_model.update_model()

        # Simulate the found values for plotting
        #est_temp_stue_m = model.solve(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fireplace_power, data_energy, x_sol[1], x_sol[0])
        est_temp_stue_m = ss_model.solve_continuous(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fireplace_power, data_energy)

        # Check that the discrete model behaves
        #est_temp_stue_md = model.solve_disc(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fireplace_power, data_energy, x_sol[1], x_sol[0])
        est_temp_stue_md = ss_model.solve_discrete(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fireplace_power, data_energy)


    # Estimate using model and known distrubances
    est_temp_stue_1 = model.solve(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fireplace_power, data_energy, k, ha)
    est_temp_stue_2 = model.solve(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fireplace_power, data_energy, k, 300)
    est_temp_stue_3 = model.solve_disc(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fireplace_power, data_energy, k, 310)


    #print("Cost 1: ", cost([ha, k]))
    #print("Cost 2: ", cost([300, k]))


    has = [111]
    ks = [1333333*4.3]
    fs = [2000]

    # Use this to plog for comibnations of ha, k and fireplace power
    #has = np.linspace(100, 200, 10)
    #ks = np.linspace(500000, 3000000, 10)
    #fs = np.linspace(500, 3000, 10)

    series_search = []
    for hai in has:
        for ki in ks:
            for fi in fs:
                series_search.append((model.solve(data_temp_stue[0], time_series, data_temp_ute, data_fireplace*fi, data_energy, ki, hai), "ha: {}, k: {} f: {}".format(hai, ki, fi)))


    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time_series, data_temp_stue + CELCIUS, label='Målt temp stue')
    ax1.plot(time_series, data_temp_ute + CELCIUS, label='Målt temp ute')
    #ax2.plot(time_series, data_energy, label='Energy')
    ax1.plot(time_series, data_fireplace*20, label='Peis')
    #ax1.plot(time_series, est_temp_stue_1 + CELCIUS, label='Stue temp estimated 1')
    #ax1.plot(time_series, est_temp_stue_2 + CELCIUS, label='Stue temp estimated 2')
    #ax1.plot(time_series, est_temp_stue_3 + CELCIUS, label='Stue temp estimated 3')

    for ss in series_search:
        ax1.plot(time_series, ss[0] + CELCIUS, label=ss[1])

    if est_temp_stue_m is not None:
        #ax1.plot(time_series, est_temp_stue_m + CELCIUS, label='Stue temp estimated minimized')
        ax1.plot(time_series, est_temp_stue_m + CELCIUS, label='Fitted LTI model continuous')
        ax1.plot(time_series, est_temp_stue_md + CELCIUS, label='Fitted LTI model discrete')
    ax1.xaxis.set_major_formatter(lambda x, pos: "{:.0f}h".format(x / 3600.0))
    plt.xlabel('t')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, 8000])

    ax1.set_xlabel('t')
    ax1.set_ylabel('Tempearture [C]')
    ax2.set_ylabel('Gasspeis [W]')
    plt.title("Model identification")

    plt.grid()
    plt.show()


