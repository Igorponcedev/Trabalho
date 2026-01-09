import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Previsão de Obesidade", layout="centered")

# =========================
# CARREGAR MODELOS
# =========================

@st.cache_resource
def load_artifacts():
    model = joblib.load("modelo_obesidade.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    expected_columns = joblib.load("expected_columns.pkl")
    return model, label_encoder, expected_columns


model, label_encoder, expected_columns = load_artifacts()

# =========================
# INTERFACE
# =========================

st.title("🧠 Previsão de Nível de Obesidade")
st.write("Preencha os dados abaixo para prever o nível de obesidade:")

with st.form("form_obesidade"):

    Age = st.number_input("Idade", 1, 120, 25)
    Height = st.number_input("Altura (m)", 1.0, 2.5, 1.70)
    Weight = st.number_input("Peso (kg)", 30.0, 300.0, 70.0)

    Gender = st.selectbox("Gênero", ["Male", "Female"])
    family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
    FAVC = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
    SMOKE = st.selectbox("Fuma?", ["yes", "no"])
    SCC = st.selectbox("Monitora calorias?", ["yes", "no"])

    FCVC = st.slider("Consumo de vegetais", 1.0, 3.0, 2.0)
    NCP = st.slider("Número de refeições por dia", 1.0, 4.0, 3.0)
    CH2O = st.slider("Consumo de água", 1.0, 3.0, 2.0)
    FAF = st.slider("Atividade física", 0.0, 3.0, 1.0)
    TUE = st.slider("Uso de tecnologia", 0.0, 3.0, 1.0)

    CAEC = st.selectbox("Come entre refeições?", ["no", "Sometimes", "Frequently", "Always"])
    CALC = st.selectbox("Consome álcool?", ["no", "Sometimes", "Frequently", "Always"])
    MTRANS = st.selectbox(
        "Meio de transporte",
        ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
    )

    submit = st.form_submit_button("🔍 Prever")

# =========================
# PREVISÃO
# =========================

if submit:

    data = {
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "Gender": Gender,
        "family_history": family_history,
        "FAVC": FAVC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "CAEC": CAEC,
        "CALC": CALC,
        "MTRANS": MTRANS,
    }

    df_input = pd.DataFrame([data])

    # garantir mesmas colunas e ordem do treino
    df_input = df_input.reindex(columns=expected_columns)

    pred = model.predict(df_input)
    classe = label_encoder.inverse_transform(pred)[0]

    st.success(f"🎯 Nível previsto: **{classe}**")