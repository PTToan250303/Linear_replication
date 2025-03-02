import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from streamlit_option_menu import option_menu
import Lythuyet as Lythuyet

def Classification():
    # Định dạng tiêu đề
    st.markdown("""
        <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #4682B4;  /* Màu xanh nước biển nhạt */
            margin-top: 50px;
        }
        .subtitle {
            font-size: 24px;
            text-align: center;
            color: #4A4A4A;  /* Màu xám đậm */
        }
        <hr>
        </style>
        <div class="title">Khái phá dữ liệu</div>
        <div class="subtitle">Data Processing</div>
        <hr>
    """, unsafe_allow_html=True)

    # Cho phép người dùng tải nhiều file
    uploaded_files = st.file_uploader("📥Chọn các file dataset", accept_multiple_files=True)

    datasets = {}
    error_files = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                datasets[uploaded_file.name] = df
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
                datasets[uploaded_file.name] = df
            else:
                error_files.append(uploaded_file.name)

        # Hiển thị thông báo lỗi kèm biểu tượng cảnh báo
        if error_files:
            st.warning("⚠️Các tệp không được chấp nhận: .Rhistory, body.txt, CrossTable, DocFile, file_example_XLS_100.xls. Vui lòng chỉ tải lên các tệp .csv hoặc .xlsx.⚠️")

    # Chỉ hiển thị thanh điều hướng khi đã có file được tải lên hợp lệ
    if datasets and not error_files:
        # === Tạo Tabs ===
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📘 Lý thuyết Random Forest", 
            "🗄️ Data",
            "📊 Xử lý dữ liệu",
            "⚙️ Huấn luyện", 
            "💡 Demo",
            "📝 MLflow"
        ])

        with tab1:
            Lythuyet.ly_thuyet_Random_Forest()

        with tab2:
           Lythuyet.ly_thuyet_Random_Forest()
            
        with tab3:
            Lythuyet.ly_thuyet_Random_Forest()

        with tab4:
            Lythuyet.ly_thuyet_Random_Forest()

        with tab5:
            Lythuyet.ly_thuyet_Random_Forest()
        
        with tab6:
            # Nút bấm "Dự đoán với MLflow"
            Lythuyet.ly_thuyet_Random_Forest()
    else:
        st.write("Vui lòng tải lên ít nhất một file dữ liệu để bắt đầu.")

if __name__ == "__main__":
    Classification()
