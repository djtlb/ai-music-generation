#!/usr/bin/env python3
"""
CLI script for evaluating critic models and DPO finetuning results.

Usage:
  python evaluate_model.py --critic_checkpoint critic_best.pth --test_data validation.csv --audio_dir ./audio
  
  # Create validation playlist
  python evaluate_model.py --create_playlist --test_data validation_playlist.csv
  
  # Compare before/after DPO
  python evaluate_model.py --critic_checkpoint critic_best.pth --test_data validation.csv --audio_dir ./audio --before_dpo before.pth --after_dpo after.pth
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.critic.evaluate import main

if __name__ == "__main__":
    main()