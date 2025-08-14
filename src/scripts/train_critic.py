#!/usr/bin/env python3
"""
CLI script for training the critic reward model.

Usage:
  python train_critic.py --data_csv preference_data.csv --audio_dir ./audio_clips --epochs 100
  
  # Create mock data for testing
  python train_critic.py --mock_data --data_csv mock_preferences.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.critic.train import main

if __name__ == "__main__":
    main()