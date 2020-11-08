
import numpy as np

def relu(x):
  return x * (x > 0)
    
def avg_pool(x, p):
  H, W, C = np.shape(x)
  x = np.reshape(x, (H // p, p, W // p, p, C))
  x = np.transpose(x, (0, 2, 1, 3, 4))
  x = np.mean(x, axis=(2, 3))
  return x

def max_pool(x, k, s):
    p = k // 2

    xh, xw, xc = np.shape(x)
    yh, yw, yc = (xh // s), (xw // s), xc

    x = np.pad(array=x, pad_width=[[p,p], [p,p], [0,0]], mode='constant')
    y = np.zeros(shape=(yh, yw, yc))
    for h in range(yh):
        for w in range(yw):
            y[h, w, :] = np.max(x[h*s:(h*s+k), w*s:(w*s+k), :], axis=(0, 1))

    return y

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


def conv_input_length(output_length, filter_size, padding, stride):
  """Determines input length of a convolution given output length.

  Arguments:
      output_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.

  Returns:
      The input length (integer).
  """
  if output_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  if padding == 'same':
    pad = filter_size // 2
  elif padding == 'valid':
    pad = 0
  elif padding == 'full':
    pad = filter_size - 1
  return (output_length - 1) * stride - 2 * pad + filter_size


def get_pad(padding, filter_size):
  if padding == 'same':
      pad = filter_size // 2
  elif padding == 'valid':
      pad = 0
  elif padding == 'full':
      pad = filter_size - 1
  else:
      assert(False)
  return pad

