@echo off
pushd %~dp0\..
echo ============================================
echo  AI Personalized Learning System â€” Startup
echo ============================================

:: 1. Install dependencies
echo.
echo [1/3] Installing Python packages...
pip install -r requirements.txt

:: 2. Generate dataset
echo.
echo [2/3] Generating dataset...
python ml/data/generate_dataset.py

:: 3. Train models
echo.
echo [3/3] Training ML models...
python ml/training/train_model.py

echo.
echo ============================================
echo  Setup complete!
echo.
echo  To start the backend:
echo    uvicorn api.main:app --reload
echo.
echo  To start the frontend (new terminal):
echo    streamlit run frontend/legacy/app.py
echo ============================================
pause
popd


