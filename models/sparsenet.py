# ----------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2019 Numenta, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------

import torch.nn as nn
from nupic.torch.modules import KWinners, KWinners2d, SparseWeights, Flatten



class SparseCNN(nn.Module):

  def __init__(self, out_dim, in_channel, img_sz, n, k, k_inference_factor,
               boost_strength, boost_strength_factor, weight_sparsity,
               out_channels, kernel_size, stride, padding, cnn_k):
    super(SparseCNN, self).__init__()
    self.in_dim = in_channel * img_sz * img_sz

    # Compute Flatten CNN output len
    maxpool = []
    maxpool.append(((img_sz + 2 * padding[0] - kernel_size[0]) // stride[0] + 1) // 2)
    maxpool.append(((maxpool[0] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1) // 2)

    cnn_output_len = [maxpool[i] * maxpool[i] * out_channels[i]
                      for i in range(len(maxpool))]

    # Create sparseCNN2 model
    self.linear = nn.Sequential(
      nn.Conv2d(in_channels=in_channel, out_channels=out_channels[0],
                kernel_size=kernel_size[0], stride=stride[0],
                padding=padding[0]),
      KWinners2d(n=cnn_output_len[0], k=cnn_k[0],
                 channels=out_channels[0],
                 kInferenceFactor=k_inference_factor,
                 boostStrength=boost_strength,
                 boostStrengthFactor=boost_strength_factor),
      nn.MaxPool2d(kernel_size=2),

      nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1],
                kernel_size=kernel_size[1], stride=stride[1],
                padding=padding[1]),
      KWinners2d(n=cnn_output_len[1], k=cnn_k[1],
                 channels=out_channels[1],
                 kInferenceFactor=k_inference_factor,
                 boostStrength=boost_strength,
                 boostStrengthFactor=boost_strength_factor),
      nn.MaxPool2d(kernel_size=2),

      Flatten(),

      SparseWeights(nn.Linear(cnn_output_len[1], n), weight_sparsity),
      KWinners(n=n, k=k, kInferenceFactor=k_inference_factor,
               boostStrength=boost_strength,
               boostStrengthFactor=boost_strength_factor)
    )
    self.last = nn.Linear(n, out_dim)


  def features(self, x):
    x = self.linear(x)
    return x


  def logits(self, x):
    x = self.last(x)
    return x


  def forward(self, x):
    x = self.features(x)
    x = self.logits(x)
    return x



class SparseMLP(nn.Module):

  def __init__(self, out_dim, in_channel, img_sz, hidden_dim,
               weight_sparsity, k, k_inference_factor, boost_strength,
               boost_strength_factor):
    super(SparseMLP, self).__init__()
    self.in_dim = in_channel * img_sz * img_sz
    self.linear = nn.Sequential(
      SparseWeights(nn.Linear(self.in_dim, hidden_dim), weight_sparsity),
      KWinners(n=hidden_dim, k=k[0], kInferenceFactor=k_inference_factor,
               boostStrength=boost_strength,
               boostStrengthFactor=boost_strength_factor),
      SparseWeights(nn.Linear(hidden_dim, hidden_dim), weight_sparsity),
      KWinners(n=hidden_dim, k=k[1], kInferenceFactor=k_inference_factor,
               boostStrength=boost_strength,
               boostStrengthFactor=boost_strength_factor)
    )
    self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task


  def features(self, x):
    x = self.linear(x.view(-1, self.in_dim))
    return x


  def logits(self, x):
    x = self.last(x)
    return x


  def forward(self, x):
    x = self.features(x)
    x = self.logits(x)
    return x



def SparseCNN2():
  return SparseCNN(out_dim=10, in_channel=1, img_sz=32, n=300, k=50,
                   k_inference_factor=1.5, boost_strength=0.0,
                   boost_strength_factor=1.0, weight_sparsity=0.3,
                   out_channels=(30, 40), kernel_size=(5, 5), stride=(1, 1),
                   padding=(0, 0), cnn_k=(400, 400))



def SparseMLP400():
  return SparseMLP(out_dim=10, in_channel=1, img_sz=32, hidden_dim=400,
                   weight_sparsity=0.3, k=(80, 80), k_inference_factor=1.5,
                   boost_strength=0.0, boost_strength_factor=1.0)



def SparseMLP1000():
  return SparseMLP(out_dim=10, in_channel=1, img_sz=32, hidden_dim=1000,
                   weight_sparsity=0.3, k=(200, 200), k_inference_factor=1.5,
                   boost_strength=0.0, boost_strength_factor=1.0)
