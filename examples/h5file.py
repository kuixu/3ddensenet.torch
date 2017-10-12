import numpy as np
import h5py, sys
filename=sys.argv[1]
a=h5py.File(filename,"r")
print np.unique(a['label'])
print a['data'].shape
