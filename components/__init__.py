# Components package for aiflow pipeline orchestration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decorator import component

__all__ = ['component']