# app.py para Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Configuração da página
st.set_page_config(
    page_title="Preditor de Nível de Obesidade",
    page_icon="🏥",
    layout="wide"
)

# Título
st.title("🏥 Preditor de Nível de Obesidade")
st.markdown("""
Esta aplicação utiliza machine learning para prever o nível de obesidade com base em características físicas e hábitos.
""")

# Carregar modelo e encoders
@st.cache_resource
def load_artifacts():
    """Carrega o modelo e outros arquivos necessários"""
    try:
        model = joblib.load("modelo_obesidade.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        expected_columns = joblib.load("expected_columns.pkl")
        categories = joblib.load("categories.pkl")
        return model, label_encoder, expected_columns, categories
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None, None, None

# Carregar artefatos
model, label_encoder, expected_columns, categories = load_artifacts()

if model is None:
    st.stop()

# Sidebar para entrada de dados
st.sidebar.header("📊 Informações do Paciente")

# Mapeamento de labels
obesity_labels = {
    0: "Insufficient_Weight",
    1: "Normal_Weight", 
    2: "Overweight_Level_I",
    3: "Obesity_Type_I",
    4: "Obesity_Type_II",
    5: "Obesity_Type_III"
}

# Funções auxiliares
def calculate_bmi(weight, height):
    """Calcula o BMI"""
    return weight / (height ** 2)

def classify_bmi(bmi):
    """Classifica o BMI em categorias"""
    if bmi < 18.5:
        return "Baixo peso"
    elif bmi < 25:
        return "Peso normal"
    elif bmi < 30:
        return "Sobrepeso Nível I"
    elif bmi < 35:
        return "Obesidade Tipo I"
    elif bmi < 40:
        return "Obesidade Tipo II"
    else:
        return "Obesidade Tipo III"

# Formulário de entrada
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Idade", min_value=14, max_value=100, value=30)
        height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
        gender = st.selectbox("Gênero", ["Male", "Female"])
        family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
        favc = st.selectbox("Consumo frequente de alimentos calóricos", ["yes", "no"])
    
    with col2:
        fvc = st.slider("Frequência de consumo de vegetais (1-3)", 1.0, 3.0, 2.0, step=0.1)
        ncp = st.slider("Número de refeições principais (1-4)", 1.0, 4.0, 3.0, step=0.1)
        caec = st.selectbox("Consumo de alimentos entre refeições", categories["CAEC"])
        smoke = st.selectbox("Fuma?", ["yes", "no"])
        ch2o = st.slider("Consumo diário de água (L)", 0.5, 3.0, 1.5, step=0.1)
        scc = st.selectbox("Monitora calorias consumidas?", ["yes", "no"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        faf = st.slider("Frequência de atividade física (0-3)", 0.0, 3.0, 1.0, step=0.1)
        tue = st.slider("Tempo usando dispositivos eletrônicos (0-2)", 0.0, 2.0, 1.0, step=0.1)
    
    with col4:
        calc = st.selectbox("Consumo de álcool", categories["CALC"])
        mtrans = st.selectbox("Meio de transporte principal", 
                             ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
    
    submitted = st.form_submit_button("🔍 Prever Nível de Obesidade")

# Processar quando o formulário for enviado
if submitted:
    try:
        # Calcular BMI
        bmi = calculate_bmi(weight, height)
        bmi_category = classify_bmi(bmi)
        
        # Criar DataFrame com os dados
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Age": float(age),
            "Height": height,
            "Weight": weight,
            "family_history": family_history,
            "FAVC": favc,
            "FCVC": float(fvc),
            "NCP": float(ncp),
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": float(ch2o),
            "SCC": scc,
            "FAF": float(faf),
            "TUE": float(tue),
            "CALC": calc,
            "MTRANS": mtrans
        }])
        
        # Garantir que as colunas estão na ordem correta
        input_data = input_data[expected_columns]
        
        # Fazer predição
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Decodificar a predição
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        
        # Traduzir para português
        label_translation = {
            "Insufficient_Weight": "Baixo Peso",
            "Normal_Weight": "Peso Normal",
            "Overweight_Level_I": "Sobrepeso Nível I",
            "Obesity_Type_I": "Obesidade Tipo I",
            "Obesity_Type_II": "Obesidade Tipo II",
            "Obesity_Type_III": "Obesidade Tipo III"
        }
        
        translated_label = label_translation.get(prediction_label, prediction_label)
        
        # Exibir resultados
        st.success("✅ Predição concluída!")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BMI Calculado", f"{bmi:.2f}")
        
        with col2:
            st.metric("Categoria BMI", bmi_category)
        
        with col3:
            # Encontrar a maior probabilidade
            max_prob = np.max(prediction_proba) * 100
            st.metric("Confiança da Predição", f"{max_prob:.1f}%")
        
        # Resultado principal
        st.subheader(f"📋 Nível de Obesidade Previsto: **{translated_label}**")
        
        # Gráfico de probabilidades
        st.subheader("📊 Probabilidades por Categoria")
        
        prob_df = pd.DataFrame({
            "Categoria": [label_translation.get(lbl, lbl) for lbl in label_encoder.classes_],
            "Probabilidade (%)": (prediction_proba * 100).round(1)
        }).sort_values("Probabilidade (%)", ascending=False)
        
        st.bar_chart(prob_df.set_index("Categoria"))
        
        # Recomendações baseadas no resultado
        st.subheader("💡 Recomendações")
        
        recommendations = {
            "Baixo Peso": """
            - Aumentar consumo calórico de forma saudável
            - Incluir proteínas magras e carboidratos complexos
            - Consultar nutricionista para plano alimentar
            - Exercícios de força para ganho muscular
            """,
            "Peso Normal": """
            - Manter hábitos alimentares saudáveis
            - Continuar com atividade física regular
            - Monitorar peso mensalmente
            - Manter hidratação adequada
            """,
            "Sobrepeso Nível I": """
            - Reduzir calorias em 200-300 por dia
            - Aumentar atividade física para 150 min/semana
            - Reduzir alimentos processados e açúcares
            - Acompanhar ingestão alimentar
            """,
            "Obesidade Tipo I": """
            - Consultar médico e nutricionista
            - Redução calórica supervisionada
            - Exercícios aeróbicos 30 min/dia, 5x/semana
            - Monitorar progresso semanalmente
            """,
            "Obesidade Tipo II": """
            - Acompanhamento médico obrigatório
            - Plano alimentar personalizado
            - Atividade física supervisionada
            - Considerar acompanhamento psicológico
            """,
            "Obesidade Tipo III": """
            - Intervenção médica imediata
            - Tratamento multidisciplinar
            - Possível indicação cirúrgica
            - Acompanhamento intensivo
            """
        }
        
        st.info(recommendations.get(translated_label, "Consulte um profissional de saúde."))
        
        # Download dos dados
        st.download_button(
            label="📥 Baixar Relatório",
            data=input_data.to_csv(index=False),
            file_name="dados_paciente.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Erro ao processar predição: {e}")
        st.info("Verifique se todos os campos foram preenchidos corretamente.")

# Informações adicionais na sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
**Sobre o Modelo:**
- Treinado com múltiplos algoritmos de ML
- Validação cruzada de 5 folds
- Precisão: ~90% nos dados de teste

**Classificações BMI:**
- < 18.5: Baixo peso
- 18.5-25: Normal
- 25-30: Sobrepeso I
- 30-35: Obesidade I
- 35-40: Obesidade II
- > 40: Obesidade III
""")

# Rodapé
st.markdown("---")
st.caption("⚠️ Esta ferramenta é para fins educacionais. Consulte um profissional de saúde para diagnóstico médico.")