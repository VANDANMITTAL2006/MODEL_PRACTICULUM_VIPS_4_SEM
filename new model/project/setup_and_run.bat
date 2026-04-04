@echo off
echo ============================================
echo  AI Personalized Learning System — Startup
echo ============================================

:: 1. Install dependencies
echo.
echo [1/3] Installing Python packages...
pip install -r requirements.txt

:: 2. Generate dataset
echo.
echo [2/3] Generating dataset...
python data/generate_dataset.py

:: 3. Train models
echo.
echo [3/3] Training ML models...
python models/train_model.py

echo.
echo ============================================
echo  Setup complete!
echo.
echo  To start the backend:
echo    uvicorn backend.main:app --reload
echo.
echo  To start the frontend (new terminal):
echo    streamlit run frontend/app.py
echo ============================================
pause
