import streamlit as st
import sys
from pathlib import Path

# Add deployment directory to path
deployment_path = Path(__file__).parent / "deployment"
sys.path.insert(0, str(deployment_path))

# Run the main Streamlit app
exec(open(deployment_path / "streamlit_app.py").read())
