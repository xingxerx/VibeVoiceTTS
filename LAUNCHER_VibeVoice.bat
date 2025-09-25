@echo off
echo Starting VibeVoice Demo...
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at .venv\Scripts\activate.bat
    echo Please make sure the virtual environment is set up correctly.
    echo.
    pause
    exit /b 1
)


REM Activate virtual environment
echo Activating virtual environment...
CALL .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)

REM Check if gradio_demo.py exists
if not exist "demo\gradio_demo.py" (
    echo ERROR: demo\gradio_demo.py not found
    echo Please make sure you're running this from the correct directory.
    echo.
    pause
    exit /b 1
)


REM Launch the demo
echo Launching VibeVoice Gradio Demo...
.venv\Scripts\python.exe demo\gradio_demo.py
if errorlevel 1 (
    echo.
    echo ERROR: Demo failed to start. Check the error messages above.
    echo.
    pause
    exit /b 1
)

REM List of possible arguments
REM 		--model_path        		default="./models"        										"Path to the VibeVoice model directory"
REM 		--device        			default="cuda" if torch.cuda.is_available() else "cpu"        	"Device for inference"
REM 		--inference_steps           default=10        												"Number of inference steps for DDPM (not exposed to users)"
REM 		--share        				action="store_true"        										"Share the demo publicly via Gradio"
REM 		--port               		default=7860        											"Port to run the demo on"
