import concurrent.futures
import time

start = time.perf_counter()

def do_somthing(i):
    print('Sleeping 1 second ...')
    time.sleep(1)
    return 'done sleep'

with concurrent.futures.ProcessPoolExecutor() as executor:
    secs = [5,4,3,2,1]
    results = [executor.submit(do_somthing,1) for _ in range(10)]

    for f in concurrent.futures.as_completed(results):
        print(f.result())


# p1 =multiprocessing.Process(target=do_somthing)
# p2 =multiprocessing.Process(target=do_somthing)

# p1.start()
# p2.start() 

# p1.join()
# p2.join()

finish = time.perf_counter()

print(f'Finished in {round(finish-start,2)}second(s)')