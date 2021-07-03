import matplotlib.pyplot as plt
import numpy as np
from psf_utils import PSF, Quantity

import matplotlib.pyplot as plt
import os
from pytictoc import TicToc


def linear(sample, eval):

    sampleSize = np.size(sample)
    resultSize = np.size(eval)

    # interval should be integer.
    interval = int((resultSize - 1) / (sampleSize - 1))
    
    result = np.zeros(resultSize)    

    for idx in range(sampleSize):
        result[interval * idx] = sample[idx]
        
    for i in range(sampleSize - 1):
        # idx for result array. interval * sampleidx
        idx = interval * i
        # slope = f(x+interval) - f(x)  /  interval                   
        slope = (result[idx + interval] - result[idx])/interval
        for j in range(idx + 1, idx + interval):
            result[j] = slope * (j - idx) + sample[i]
        
    if idx + interval != resultSize - 1:
        for j in range(idx+interval+1, resultSize):
            result[j] = slope * (j - (idx + interval)) + sample[i+1]

    return result

def lagrange(xAxisSample, xAxisEval, sample, eval):

    sampleSize = np.size(sample)
    resultSize = np.size(eval)

    # interval should be integer.
    interval = int((resultSize - 1) / (sampleSize - 1))
    
    result = np.zeros(resultSize)        

    for i in range(resultSize):

        # xidx : unused x sample in lagrange term
        xidx = 0   
        for iteration in range(sampleSize):         
            tmp = sample[xidx]
            for point in xAxisSample:      
                if point != xAxisSample[xidx]:                    
                        tmp *= (xAxisEval[i] - point)     
                        tmp /= (xAxisSample[xidx] - point)

            result[i] += tmp
            xidx += 1

    print(result)
    return result


            


def avgError(result, eval):
    size = np.size(result)
    err = np.zeros(size)

    for i in range(size):
        err[i] = abs(result[i] - eval[i]) / eval[i]
   
    return 100 * np.average(err)


    
    





# Read file from spectre scs file.
path_dir = './'
file_list = os.listdir(path_dir)


# Reads the sample data and the data for evaluation
for data in file_list:
    if '.dc' in data and '.cache' not in data:
        psf = PSF(data)
        with Quantity.prefs(map_sf=Quantity.map_sf_to_greek):
            for signal in sorted(psf.all_signals(), key=lambda s: s.name):
                name = f'{signal.access}({signal.name})'

                # X axis
                if name == 'V(drain)':
                    if 'sample' in data:
                        xAxisSample = signal.ordinate
                    else:
                        xAxisEval = signal.ordinate

                # Data
                elif name == 'I(vs:p)' :
                    if 'sample' in data:
                        sampleData = signal.ordinate
                    else:
                        evalData = signal.ordinate

# sample     :      xAxisSample - sampleData
# evaluation :      xAxisEval   - evalData


# Tictoc class for measuring turnaround time
tat = TicToc()


# Interpolation function is wrapped by tictoc.
tat.tic()
# result = linear(sampleData, evalData)
result = lagrange(xAxisSample, xAxisEval, sampleData, evalData)
tat.toc()

err = avgError(result, evalData)
print("average error : ", err,"%")

plt.figure()
plt.scatter(xAxisSample, sampleData, s=1)
plt.title('Samples')

plt.figure()
plt.plot(xAxisEval, result)
plt.title('Interpolation')

plt.figure()
plt.scatter(xAxisEval, evalData, s=1)
plt.title('Target')

plt.show()



