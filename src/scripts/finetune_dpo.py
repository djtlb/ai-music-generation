#!/usr/bin/env python3
"""
CLI script for DPO finetuning of music generation models.

Usage:
  python finetune_dpo.py --data_path dpo_pairs.json --critic_checkpoint critic_best.pth --epochs 5
  
  # Create mock DPO data for testing
  python finetune_dpo.py --mock_data --data_path ./mock_dpo_data/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.finetune.dpo import main

if __name__ == "__main__":
    main()