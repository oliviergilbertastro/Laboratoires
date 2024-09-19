import numpy as np
import os
import sys
import parse
code_cours = os.path.dirname(os.path.realpath(__file__))[-4:]
parent_dir = parse.parse("{}\PHY-"+code_cours, os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(parent_dir)
from utils import *
print_color('hello word')