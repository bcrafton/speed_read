
import numpy as np
import tensorflow as tf

###########################################

# conv_output_length not relevant
def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.

  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding == 'same':
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride

###########################################

# this function passes the filter and each of the 32x32 patches to conv_kernel
# conv_kernel is what we care about
def conv(x, f, bpa, bpw, rpr, adc):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, 4, 'valid', 1)
    Wo = conv_output_length(Hi, 4, 'valid', 1)

    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))
    y = np.zeros(shape=(Ho, Wo, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h:(h+4), w:(w+4), :], -1)
            assert(np.prod(np.shape(patch)) == np.shape(f_matrix)[0])
            y[h, w, :] = conv_kernel(patch, f_matrix, bpa, bpw, rpr, adc)

    return y

###########################################

def conv_kernel(patch, f, bpa, bpw, rpr, adc):
    def ref(patch, f):
        y = patch @ f
        assert(np.all(np.absolute(y) < 2 ** 15))
        y = y * (y > 0)
        y = y.astype(int)
        return y

    y = 0
    for xb in range(bpa):
        patch_xb = np.bitwise_and(np.right_shift(patch.astype(int), xb), 1)

        nrow = len(patch_xb)
        for r1 in range(0, nrow, rpr):
            r2 = min(nrow, r1 + rpr)
            offset_xb = np.sum(patch_xb[r1:r2]) << (xb + (bpw - 1))
            offset_xb_shift = offset_xb >> (rpr // adc - 1)

            for wb in range(bpw):
                f_wb = np.bitwise_and(np.right_shift(f.astype(int), wb), 1)
                
                dot = patch_xb[r1:r2] @ f_wb[r1:r2]                
                dot_shift = dot >> (rpr // adc - 1)                
                y = y + np.left_shift(dot_shift.astype(int), xb + wb)
            
            y = y - offset_xb_shift

    assert(np.all(y < 2 ** 15))
    y = y * (y > 0)
    y = y.astype(int)
    
    y_ref = ref(patch, f - (1 << (bpw - 1)))
    print (y); print (y_ref)
    assert(np.all(y == y_ref))
    return y
    
###########################################

bpa=8
bpw=4
rpr=16
adc=8

# not necessarily rules, but whats required for code to work.
assert(rpr >= adc)
assert(rpr % adc == 0)

values = np.array(range(1, 2 ** bpw))
f1 = np.random.choice(a=values, size=[4,4,3,32], replace=True)

###########################################

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
assert(np.shape(x_train) == (50000, 32, 32, 3))

x_train = x_train / 255 * 63
x_train = x_train.astype(int)

x = x_train[0]
y1 = conv(x, f1, bpa=bpa, bpw=bpw, rpr=rpr, adc=adc)

###########################################





























