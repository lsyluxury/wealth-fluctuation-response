#!/usr/bin/env python3
"""Compatibility entry: delegates to the approved figure builder."""
from pathlib import Path
import subprocess,sys
if __name__=="__main__":
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"build_final_figures.py"),*sys.argv[1:]],check=True)
