import multiprocessing as mp
import random
import string
import time

random.seed(123)

# Define an output queue
output = mp.Queue()

# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    j = 0
    while(j< 3*(10**6)):
        rand_str = ''.join(random.choice(
                            string.ascii_lowercase
                           + string.ascii_uppercase
                            + string.digits)
                    for i in range(length))
        j+=1
    output.put(rand_str)


# Setup a list of processes that we want to run

processes = [mp.Process(target=rand_string, args=(5, output)) for x in range(90)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]

print(results)
