#name:			ggmbes
#created:		July 2017
#by:			p.kennedy@guardiangeomatics.com
#description:	python module to represent MBES data so we can QC, compute and merge.

import os.path
import struct
import pprint
import time
import datetime
import math
import random
from datetime import datetime
from datetime import timedelta
from statistics import mean
import numpy as np

###############################################################################
class GGPING:
	'''used to hold the metadata associated with a ping of data.'''
	def __init__(self):
		self.timestamp			= 0
		self.longitude 			= 0
		self.latitude 			= 0
		self.ellipsoidalheight 	= 0
		self.heading		 	= 0
		self.pitch			 	= 0
		self.roll			 	= 0
		self.heave			 	= 0
		self.tidecorrector	 	= 0

