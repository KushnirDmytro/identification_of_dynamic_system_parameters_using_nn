def init_logger(config):
    if not os.path.exists(config['exp_path']):
        os.mkdir(config['exp_path'])
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    if not os.path.exists(config['exp_path'] + 'logs/'):
        os.mkdir(config['exp_path'] + 'logs/')
    logfile = f"{config['exp_path']}logs/exp_{st}.log"
    logging.basicConfig(filename=logfile,
                        level=logging.DEBUG)