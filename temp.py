import math
import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from python_speech_features import logfbank
from python_speech_features import mfcc