@echo off

:: 1. Start Backends (in new windows)
echo Starting GNN Backend Server...
start cmd /k "cd GNN\smart-contract-backend & python app_gnn.py"

echo Starting RNN Backend Server...
start cmd /k "cd RNN & python app_rnn.py"

:: 2. Start Frontend (in a new window)
echo Starting Frontend Server (npm run dev)...
start cmd /k "cd GNN\smart-contract-frontend\frontend & npm run dev"

:: 3. Wait for Frontend to initialize
echo Waiting 5 seconds for the frontend server to start up...
timeout /t 5 /nobreak > nul

:: 4. Automatically open the browser
echo Redirecting to frontend at localhost:5173...
start http://localhost:5173

echo All components launched.
pause > nul