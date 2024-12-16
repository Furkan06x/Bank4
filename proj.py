# -*- coding: utf-8 -*-
"""
Created on 13/12/2024

@author: 

* Furkan Özbek
* Emir Alparslan Dikici
* Berat Yüceldi
* Zeynep Ece Aşkın
"""

import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Sayfa yapılandırması
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="\U0001F4CA",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Stil ayarları (HTML ve CSS)
st.markdown("""
    <style>
    .stSlider > div > div > div {
        background: linear-gradient(to right, #4CAF50, #2196F3);
        border-radius: 10px;
    }
    .stRadio > label {
        font-size: 16px;
        font-weight: bold;
    }
    div.stButton > button {
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
        font-size: 16px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Model yükleme
loaded_model = pickle.load(open('bank_model.pkl.sav', 'rb'))

# Tahmin fonksiyonu
def prediction_function(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

# Ana sayfa fonksiyonu
def main():
    # Dil seçimi
    language = st.sidebar.radio("Dil Seçimi / Language", ["Türkçe", "English"])

    # Metinlerin dillerine göre ayarlanması
    if language == "Türkçe":
        titles = {
            "page_title": "Bankacılık Pazarlama Tahmin Uygulaması",
            "selection": "🔍 Sayfa Seçimi",
            "tabs": ["Tahmin", "Model Performansı", "Veri Analizi", "Hakkında"],
            "prediction": "📊 Tahmin",
            "model_performance": "🏆 Model Performansı",
            "data_analysis": "📊 Veri Analizi",
            "about": "ℹ️ Hakkında",
        }
    else:
        titles = {
            "page_title": "Bank Marketing Prediction Web App",
            "selection": "🔍 Page Selection",
            "tabs": ["Prediction", "Model Performance", "Data Analysis", "About"],
            "prediction": "📊 Prediction",
            "model_performance": "🏆 Model Performance",
            "data_analysis": "📊 Data Analysis",
            "about": "ℹ️ About",
        }

    # Sayfa Seçimi
    st.sidebar.title(titles["selection"])
    page = st.sidebar.radio("Sayfa Seç / Page Select:", titles["tabs"])

    # Tahmin Sayfası
    if page == titles["tabs"][0]:
        # Resim ekleme
        st.image("https://i.imgur.com/7rXLpz7.jpeg", use_container_width=True)

        # Sayfa başlığı ve açıklama
        st.title(titles["page_title"])
        st.write(
            "Bu uygulama, bankacılık verilerini kullanarak bir müşterinin kampanyaya katılıp katılmayacağını tahmin eder."
            if language == "Türkçe"
            else "This application predicts whether a customer will subscribe to a bank campaign using data."
        )

        # Parametre giriş alanları
        st.header(
            "🔍 Tahmin İçin Gerekli Parametreleri Girin:" if language == "Türkçe" else "🔍 Enter Parameters for Prediction:"
        )
        duration = st.slider(
            "🕒 Süre" if language == "Türkçe" else "🕒 Duration",
            0,
            3700,
            180,
            step=10,
        )
        previous = st.slider(
            "🔄 Önceki Görüşme" if language == "Türkçe" else "🔄 Previous Contacts",
            0,
            6,
            0,
        )
        emp_var_rate = st.slider(
            "📈 Çalışma Oranı" if language == "Türkçe" else "📈 Employment Variation Rate",
            -3.5,
            1.5,
            0.1,
            step=0.1,
        )
        euribor3m = st.slider(
            "💹 3 Aylık Euribor" if language == "Türkçe" else "💹 Euribor 3 Month Rate",
            0.6,
            5.1,
            3.6,
            step=0.1,
        )
        nr_employed = st.slider(
            "👥 Çalışan Sayısı" if language == "Türkçe" else "👥 Number of Employed",
            4900.0,
            5300.0,
            5166.0,
            step=10.0,
        )

        contacted_before = st.radio(
            "📞 Daha Önce İletişim" if language == "Türkçe" else "📞 Contacted Before",
            ["Evet", "Hayır"] if language == "Türkçe" else ["Yes", "No"],
        )
        contact_cellular = st.radio(
            "📱 Hücresel İletişim" if language == "Türkçe" else "📱 Contact Cellular",
            ["Evet", "Hayır"] if language == "Türkçe" else ["Yes", "No"],
        )

        # Tahmin butonu ve sonucu
        if st.button("🚀 Tahmin Et" if language == "Türkçe" else "🚀 Predict"):
            input_data = [
                duration,
                previous,
                emp_var_rate,
                euribor3m,
                nr_employed,
                1 if contacted_before == "Evet" or contacted_before == "Yes" else 0,
                1 if contact_cellular == "Evet" or contact_cellular == "Yes" else 0,
            ]
            diagnosis = prediction_function(input_data)

            # Sonuç gösterimi
            if diagnosis == 1:
                st.success(
                    "✅ Tahmin: Müşteri kampanyaya katılabilir! (1)"
                    if language == "Türkçe"
                    else "✅ Prediction: Customer will subscribe! (1)"
                )
            else:
                st.error(
                    "❌ Tahmin: Müşteri kampanyaya katılamaz. (0)"
                    if language == "Türkçe"
                    else "❌ Prediction: Customer will not subscribe. (0)"
                )

    # Model Performansı Sayfası
    elif page == titles["tabs"][1]:
        st.title(titles["model_performance"])
        st.write(
            "Modelin doğruluk ve performans metrikleri aşağıdadır:" if language == "Türkçe" else "Model accuracy and performance metrics are as follows:"
        )
        metrics = {"Accuracy": 0.87, "F1-Score": 0.85, "Precision": 0.86, "Recall": 0.84}
        fig, ax = plt.subplots()
        ax.bar(metrics.keys(), metrics.values(), color=["#4CAF50", "#FFC107", "#FF5722", "#2196F3"])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    # Veri Analizi Sayfası
    elif page == titles["tabs"][2]:
        st.title(titles["data_analysis"])
        st.write(
            "Eğitim verisinin temel özellikleri ve analizi:" if language == "Türkçe" else "Key features and analysis of the training data:"
        )
        features = ["Duration", "Previous", "Employment Rate", "Euribor3m", "Nr Employed"]
        importances = [0.25, 0.15, 0.2, 0.3, 0.1]
        fig, ax = plt.subplots()
        ax.barh(features, importances, color="skyblue")
        ax.set_xlabel("Özellik Önemi" if language == "Türkçe" else "Feature Importance")
        st.pyplot(fig)

    # Hakkında Sayfası
    elif page == titles["tabs"][3]:
        st.title(titles["about"])
        st.write(
            """
            Bu uygulama, bankacılık verilerini analiz ederek bir müşterinin kampanyaya katılıp katılmayacağını tahmin etmek amacıyla geliştirilmiştir.
            Model, en iyi performansı sunan aşağıdaki algoritmalarla eğitilmiştir:
            - Logistic Regression
            - Random Forest Classifier
            - MLP Classifier

            **Geliştiriciler:**
            - Furkan Özbek  
            - Emir Alparslan Dikici  
            - Berat Yüceldi  
            - Zeynep Ece Aşkın  
            """
            if language == "Türkçe"
            else """
            This application was developed to predict whether a customer will subscribe to a campaign based on banking data.
            The model was trained using the following algorithms for the best performance:
            - Logistic Regression
            - Random Forest Classifier
            - MLP Classifier

            **Developers:**
            - Furkan Özbek  
            - Emir Alparslan Dikici  
            - Berat Yüceldi  
            - Zeynep Ece Aşkın  
            """
        )

# Ana fonksiyon çalıştır
if __name__ == '__main__':
    main()

# Uygulama çalıştırma: streamlit run proj.py


# to run ---- streamlit run proj.py - python3 -m streamlit run proj.py
