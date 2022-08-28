from tqdm import tqdm_gui, tqdm
from time import sleep
import uuid
from multiprocessing import Pool, freeze_support, RLock, Manager, cpu_count, Lock
import random

import warnings
warnings.filterwarnings("ignore")


def do_work(a, b, shared_array, pid):
    # print(var)
    # shared_array = var[2]
    # pid = uuid.uuid4().hex
    
    # print(shared_array)
    shared_array[pid] = True
    # p_bar = tqdm(total=1000)
    with tqdm(total=200, desc=str(a), position=pid, leave=False) as pbar:
        for i in range(0, 200):
            pbar.update(1)
            sleep(random.randint(1,15) / random.randint(200,5000))
    # p_bar.close()
    shared_array[pid] = False

if __name__ == '__main__':
    pool =Pool(cpu_count(), initargs=(RLock(),), initializer=tqdm.set_lock)
    m = Manager()
    shared_array = m.dict()

    freeze_support()
    for i in range(0, cpu_count() * 40):
        # print(i)
        sleep(.1)
        # print(len(shared_array) >= cpu_count())
        if (len(shared_array) >= cpu_count()):
            pids = [pid for pid, running in shared_array.items() if not running]
            while (len(pids) == 0):
                sleep(.1)
                # print("help")
                pids = [pid for pid, running in shared_array.items() if not running]
            # print(shared_array)
            # print(pids)

            pid = shared_array.keys().index(pids[0])
        else:
            pid = i
        # print(shared_array)
        res = pool.starmap_async(do_work, [(pid, "b", shared_array, pid)])
    
    
    while True:
        sleep(1)
        pids = [pid for pid, running in shared_array.items() if running]
        # print('running jobs:', shared_array, len(pids))
        # print(pids)
        
        if (len(pids) == 0):
            break
    
    # while(True):
    #     pass

