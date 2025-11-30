#!/bin/bash

# Exit if error
set -e

# Activate virtual env (opsional, jika pakai venv)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Gunicorn
exec gunicorn app:app --bind 0.0.0.0:${PORT:-5000}
