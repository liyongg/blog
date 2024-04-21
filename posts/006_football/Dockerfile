FROM python:3.10-slim

WORKDIR /app

COPY app.py .
COPY 2022_2023_Football_Player_Stats.csv .
COPY rf_model.pkl .
COPY rf_scaler.pkl .

RUN pip install streamlit pandas scikit-learn==1.4.0

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
