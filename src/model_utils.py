import os
import logging
import datetime
import time
import matplotlib.pyplot as plt


def init_logger(config):
    if not os.path.exists(config['exp_path']):
        os.mkdir(config['exp_path'])
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    if not os.path.exists(config['exp_path'] + '_logs/'):
        os.mkdir(config['exp_path'] + '_logs/')
    logfile = f"{config['exp_path']}logs/exp_{st}.log"
    logging.basicConfig(filename=logfile,
                        level=logging.DEBUG,
                        format='[%(asctime)s] [%(levelname)-8s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    print(f"logger inited to file [{logfile}]")
    
def plot_shared_scale(plot_requests):
    for d, l in plot_requests:
        plt.plot(d, label = l)
    plt.legend()
    plt.show()
    
def plot_multiscale(plot_requests):
    fig, ax = plt.subplots()
    # Twin the x-axis twice to make independent y-axes.
    for i, req in enumerate(plot_requests):
        d, c = req
        ax.twinx().plot(d, color = c)
    plt.show()