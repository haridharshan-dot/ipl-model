@echo off
setlocal
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
