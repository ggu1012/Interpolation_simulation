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
        slope = (result[idx + interval] - result[idx]) / interval
        for j in range(idx + 1, idx + interval):
            result[j] = slope * (j - idx) + sample[i]

    if idx + interval != resultSize - 1:
        for j in range(idx + interval + 1, resultSize):
            result[j] = slope * (j - (idx + interval)) + sample[i + 1]

    return result


def lagrange(xAxisSample, xAxisEval, sample):

    sampleSize = np.size(sample)
    resultSize = np.size(xAxisEval)

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
                    tmp *= xAxisEval[i] - point
                    tmp /= xAxisSample[xidx] - point

            result[i] += tmp
            xidx += 1

    return result


def cubicSpline(xAxisSample, xAxisEval, sample):

    splineNum = np.size(xAxisSample) - 1

    # Base: a(x-xn)^3 + b(x-xn)^2 + ...
    # x-xn = hn
    h = np.zeros(splineNum)

    abcd = np.zeros([4 * splineNum, 4 * splineNum])
    coeff = np.zeros([4 * splineNum, 1])
    const = np.zeros([4 * splineNum, 1])

    # first and second derivative
    for i in range(splineNum):
        h[i] = xAxisSample[i + 1] - xAxisSample[i]

        tmpSpl = 4 * i

        #### Traverse the sample points
        abcd[tmpSpl, tmpSpl + 3] = 1
        const[tmpSpl] = sample[i]

        abcd[tmpSpl + 1, tmpSpl] = h[i] ** 3
        abcd[tmpSpl + 1, tmpSpl + 1] = h[i] ** 2
        abcd[tmpSpl + 1, tmpSpl + 2] = h[i]
        abcd[tmpSpl + 1, tmpSpl + 3] = 1
        const[tmpSpl + 1] = sample[i + 1]

        if i < splineNum - 1:
            #### 3 an hn^2 + 2 bn hn + cn = c(n+1)
            abcd[tmpSpl + 2, tmpSpl] = 3 * (h[i] ** 2)  # a_n
            abcd[tmpSpl + 2, tmpSpl + 1] = 2 * h[i]  # b_n
            abcd[tmpSpl + 2, tmpSpl + 2] = 1  # c_n
            abcd[tmpSpl + 2, 4 * (i + 1) + 2] = -1  # c_n+1
            const[tmpSpl + 2] = 0

            #### 6 an hn + 2 bn = 2 b(n+1)
            abcd[tmpSpl + 3, tmpSpl] = 6 * h[i]  # a_n
            abcd[tmpSpl + 3, tmpSpl + 1] = 2  # b_n
            abcd[tmpSpl + 3, 4 * (i + 1) + 1] = -2  # b_n+1
            const[tmpSpl + 3] = 0

        else:

            # national boundary condition
            # S''(0) = 0 -> b0 = 0
            abcd[tmpSpl + 2, tmpSpl + 1] = 1
            const[tmpSpl + 2] = 0

            # national boundary condition
            # S''(splineNum) = 0
            #   -> 6*a_(splineNum-1)*h(splineNum-1) + 2*b_(splineNum-1) = 0
            abcd[tmpSpl + 3, tmpSpl] = 6 * h[i]  # a_n
            abcd[tmpSpl + 3, tmpSpl + 1] = 2  # b_n
            const[tmpSpl + 3] = 0

    coeff = np.dot(np.linalg.inv(abcd), const)

    resultSize = np.size(xAxisEval)
    sampleSize = np.size(xAxisSample) 
    result = np.zeros(resultSize)
    idx = 0

    for i in range(resultSize):

        if xAxisEval[i] <= xAxisSample[idx + 1]:
            tmpSpl = 4 * idx
            result[i] = (
                coeff[tmpSpl] * ((xAxisEval[i]-xAxisSample[idx]) ** 3)
                + coeff[tmpSpl + 1] * ((xAxisEval[i]-xAxisSample[idx]) ** 2)
                + coeff[tmpSpl + 2] * (xAxisEval[i]-xAxisSample[idx])
                + coeff[tmpSpl + 3]
            )

        else:
            result[i] = (
                coeff[tmpSpl] * ((xAxisEval[i]-xAxisSample[idx]) ** 3)
                + coeff[tmpSpl + 1] * ((xAxisEval[i]-xAxisSample[idx]) ** 2)
                + coeff[tmpSpl + 2] * (xAxisEval[i]-xAxisSample[idx])
                + coeff[tmpSpl + 3]
            )
            idx = idx + 1 if (idx < sampleSize - 2) else sampleSize - 2

    return result


def avgError(result, eval):
    size = np.size(result)
    err = np.zeros(size)

    for i in range(size):
        err[i] = abs(result[i] - eval[i]) / eval[i]

    return 100 * np.average(err)


# Read file from spectre scs file.
path_dir = "./"
file_list = os.listdir(path_dir)


# Reads the sample data and the data for evaluation
for data in file_list:
    if ".dc" in data and ".cache" not in data:
        psf = PSF(data)
        with Quantity.prefs(map_sf=Quantity.map_sf_to_greek):
            for signal in sorted(psf.all_signals(), key=lambda s: s.name):
                name = f"{signal.access}({signal.name})"

                # X axis
                if name == "V(drain)":
                    if "sample" in data:
                        xAxisSample = signal.ordinate
                    else:
                        xAxisEval = signal.ordinate

                # Data
                elif name == "I(vs:p)":
                    if "sample" in data:
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
# result = lagrange(xAxisSample, xAxisEval, sampleData)
result = cubicSpline(xAxisSample, xAxisEval, sampleData)
tat.toc()

err = avgError(result, evalData)
print("average error : ", err, "%")

plt.figure()
plt.scatter(xAxisSample, sampleData, s=1)
plt.title("Samples")

plt.figure()
plt.plot(xAxisEval, result)
plt.scatter(xAxisEval, evalData, s=1, color='red')
plt.legend(['Interpolation', 'Target'])
plt.grid(True)
plt.title("Interploation vs. Target")

plt.show()

