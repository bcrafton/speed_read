
import numpy as np
import time
import matplotlib.pyplot as plt
import ctypes
cim_lib = ctypes.cdll.LoadLibrary('./c.cim.so')
cim_lib.cim.restype = ctypes.c_int

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
    for xb in range(8):
      for wb in range(8):

          val = np.arange(0, 16+1)
          sign = -1 if (wb == 7) else 1
          scale = (2 ** xb) * (2 ** wb)
          
          yp = yb[:, :, xb, :, wb, :].astype(int)
          yp = np.reshape(yp, (N, NWL, BL // 8, 17, 1))
          yp = yp >= np.arange(1, np.max(yp) + 1)

          var = np.random.normal(loc=0., scale=1., size=np.shape(yp))
          var = var * 0.08 * np.sqrt(val).reshape(-1, 1)
          var = var * (yp > 0)

          yp = yp * val.reshape(-1, 1) + var
          yp = np.where((yp > 0.2) * (yp < 1.0), np.ones_like(yp), np.around(yp))

          y += np.sum(yp * sign * scale, axis=(1, 3, 4))

    return y





















