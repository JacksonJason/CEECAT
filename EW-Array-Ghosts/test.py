from tqdm import tqdm_gui
from time import sleep

import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

pool = mp.Pool(mp.cpu_count())

def do_work(var):
    for i in tqdm_gui(range(0, 1000)):
        sleep(.1)

if __name__ == '__main__':
    mp.freeze_support()
    for i in range(0, mp.cpu_count() + 1):
        res = pool.map(do_work, ("a", "b"))
    
    # while(True):
    #     pass

