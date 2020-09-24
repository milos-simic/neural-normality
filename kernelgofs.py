import kgof
import kgof.data as data
import kgof.density as density
import kgof.goftest as gof
import kgof.kernel as kernel
import kgof.util as util
import matplotlib
import matplotlib.pyplot as plt
import autograd.numpy as np
import scipy.stats as stats
import autograd.numpy as np
import numpy

def isogauss_log_den(X, d=1):
    """
    Evaluate the log density at the points (rows) in X 
    of the standard isotropic Gaussian.
    Note that the density is NOT normalized. 
    
    X: n x d nd-array
    return a length-n array
    """
    mean = np.zeros(d)
    variance = 1
    unden = -np.sum((X-mean)**2, 1)/(2.0*variance)
    return unden

def get_normal_log_den(mean, variance):
    def log_den(X, d=1):
        unden = -np.sum((X-mean)**2, 1)/(2.0*variance)
        return unden
    return log_den

# p is an UnnormalizedDensity object
#p = density.from_log_den(1, isogauss_log_den)

# dat will be fed to the test.
#dat = data.Data(X)

# J is the number of test locations (or features). Typically not larger than 10.
#J = 1

# There are many options for the optimization. 
# Almost all of them have default values. 
# Here, we will list a few to give you a sense of what you can control.
# Full options can be found in gof.GaussFSSD.optimize_locs_widths(..)
#opts = {
#    'reg': 1e-2, # regularization parameter in the optimization objective
#    'max_iter': 50, # maximum number of gradient ascent iterations
#    'tol_fun':1e-7, # termination tolerance of the objective
#}

# make sure to give tr (NOT te).
# do the optimization with the options in opts.
#V_opt, gw_opt, opt_info = gof.GaussFSSD.optimize_auto_init(p, tr, J, **opts)


# alpha = significance level of the test
#alpha = 0.01
#fssd_opt = gof.GaussFSSD(p, gw_opt, V_opt, alpha)

N01 = density.from_log_den(1, isogauss_log_den)
#J = 1

def perform_kernel_test(sample, alpha, locations=None, J=10):
    m = numpy.mean(sample)
    s = numpy.std(sample)

    sample = np.array(sample)

    if len(sample.shape) < 2:
        sample = sample.reshape(-1, 1)

    n = sample.shape[0]
    distances = [abs(sample[i] - sample[j]) for i in range(1, n) for j in range(i)]
    kernel_width = numpy.median(distances)

    if locations is None:
        locations = numpy.random.normal(m, s**2, (J, 1))

    #model = N01
    model = density.from_log_den(1, get_normal_log_den(m, s**2))
    fssd_test = gof.GaussFSSD(model, kernel_width, locations, alpha)
    dat = data.Data(sample)
    test_result = fssd_test.perform_test(dat)
    
    return test_result

def kernel_normality_test(sample, alpha, reg=1e-2, max_iter=50, tol_fun=1e-7,
    tr_proportion=0.2, J=1):
    
    test_result = perform_kernel_test(sample, alpha, None, J=J)
    return test_result['h0_rejected'] == False

def kernel_normality_test_statistic(sample, alpha=0.01, reg=1e-2, max_iter=50, tol_fun=1e-7,
    tr_proportion=0.2, J=1):
    try:
        test_result = perform_kernel_test(sample, alpha, None, J=J)
        return test_result['test_stat']
    except Exception as e:
        print(e)
        return numpy.nan
    except TypeError:
        return numpy.nan