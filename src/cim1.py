
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import ctypes
cim_lib = ctypes.cdll.LoadLibrary('./c.cim.so')
cim_lib.cim.restype = ctypes.c_int

################################################################

eps = 1e-10
inf = 1e10

def error(rpr):

    ########################################################################

    adc      = np.arange(8 + 1, dtype=np.float32)
    adc_low  = np.array([-inf, 0.2] + (adc[2:] - 0.5).tolist())
    adc_high = np.array([0.2]       + (adc[2:] - 0.5).tolist() + [inf])

    adc      =      adc.reshape(8+1, 1)
    adc_low  =  adc_low.reshape(8+1, 1)
    adc_high = adc_high.reshape(8+1, 1)

    ########################################################################
    
    s = np.arange(rpr + 1, dtype=np.float32)
    mu = s
    sd = 0.08 * np.sqrt(s)
    
    p_h = norm.cdf(adc_high, mu, np.maximum(sd, eps))
    p_l = norm.cdf(adc_low, mu, np.maximum(sd, eps))
    pe = np.clip(p_h - p_l, 0, 1)
    pe = pe / np.sum(pe, axis=0, keepdims=True)
    
    ########################################################################

    e = adc - s
    mae = np.sum(np.absolute(pe * e), axis=0)

    return mae

################################################################

def cim(xb, wb, rpr):
    N, NWL, WL, XB = np.shape(xb)
    NWL, WL, NBL, BL = np.shape(wb)

    wb = np.reshape(wb, (NWL, WL, NBL * BL))
    BL = NBL * BL

    yb = np.zeros(shape=(N, NWL, XB, BL, 17), dtype=np.uint8)

    ################################################################

    yb = np.ascontiguousarray(yb.flatten(), np.uint8)
    xb = np.ascontiguousarray(xb.flatten(), np.int8)
    wb = np.ascontiguousarray(wb.flatten(), np.int8)
    rpr_table = np.ascontiguousarray(rpr.flatten(), np.uint8)

    ################################################################

    _ = cim_lib.cim(
    ctypes.c_void_p(xb.ctypes.data), 
    ctypes.c_void_p(wb.ctypes.data), 
    ctypes.c_void_p(yb.ctypes.data), 
    ctypes.c_void_p(rpr_table.ctypes.data), 
    ctypes.c_int(N),
    ctypes.c_int(NWL),
    ctypes.c_int(WL),
    ctypes.c_int(BL))

    yb = np.reshape(yb, (N, NWL, XB, BL, 17))
    yb = np.reshape(yb, (N, NWL, XB, BL // 8, 8, 17))

    ################################################################

    y = np.zeros(shape=(N, BL // 8))
    mae = np.zeros_like(y)
    N = np.zeros_like(y)

    for xb in range(8):
      for wb in range(8):
          yp = yb[:, :, xb, :, wb, :].astype(int)
          # print (np.sum(yp, axis=(0,1,2)))
          # print (error(16))

          sign = -1 if (wb == 7) else 1
          scale = (2 ** xb) * (2 ** wb)
          val = np.arange(0, 16+1)
          y += np.sum(yp * sign * scale * val, axis=(1, 3))

          # what scales linearly ? 
          # what scales sqrt ? 
          
          # exw = error(16) * yp * val * scale
          # mae += np.sum(exw, axis=(1, 3))

          # exw = error(16) * yp * val * scale
          # mae += np.sum(exw, axis=(1, 3))
          # N += np.count_nonzero(yp, axis=(1, 3))

          # exw = error(16) * yp * val * scale
          # N = np.count_nonzero(yp, axis=(1, 3))
          # mae += np.average(exw, axis=(1, 3)) * np.sqrt(N)

          # where would we sume together ? 
          # sum(exw) / sqrt(N)
          # avg(exw) * sqrt(N)

          # LOL - why are you multiplying by val ??
          
          # exw = error(16) * yp * scale
          # N = np.count_nonzero(yp, axis=(1, 3))
          # mae += np.sum(exw, axis=(1, 3)) / np.sqrt(N)

          exw = error(16) * yp * scale
          N += np.sum(yp, axis=(1, 3))
          mae += np.sum(exw, axis=(1, 3))

    # mae = mae / np.sqrt(N)

    assert (np.shape(mae) == np.shape(y))
    mae = np.average(mae)
    print (mae)
    return y





















