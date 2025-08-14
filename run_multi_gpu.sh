#!/bin/bash

# Set environment variable to avoid tokenizers warning
export TOKENIZERS_PARALLELISM=false

# Run with accelerate using the config file
accelerate launch --config_file accelerate_config.yaml LSTMBaseline.py
