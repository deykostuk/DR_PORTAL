# DR-Vision-Mamba-Portal

This project is configured for local execution using Streamlit and the included model file.

## Run locally
1. Activate your virtual environment:
   - PowerShell: `\.\venv\Scripts\Activate.ps1`
   - Command Prompt: `\.\venv\Scripts\activate.bat`
2. Install dependencies:
   - `python -m pip install -r requirements.txt`
3. Start the app:
   - `python -m streamlit run app.py --server.port=8501 --server.address=127.0.0.1`
4. If port `8501` is busy, use a different port:
   - `python -m streamlit run app.py --server.port=8502`

## Model file
The app loads `best_model_mamba_256_final.pth` from the project root by default.

## Explainable AI (XAI) Features
This diagnostic portal provides interpretability for its predictions using two state-of-the-art XAI methods:
- **Grad-CAM:** Extracts attention heatmaps directly from the Vision Mamba architecture to highlight the exact regions of the retinal image that most heavily influenced the model's grade prediction.
- **LIME (Local Interpretable Model-agnostic Explanations):** Generates superpixel-based visual explanations using an external model to provide localized feature importance.

*Note: The LIME explanations require a secondary Keras model to function. Ensure that the `EXPLAINABLE_MODEL_PATH` environment variable points to your `.h5` model file.*

## Run Globally 
https://drapp-5lcsddn6mukokfu5zppsbg.streamlit.app/
