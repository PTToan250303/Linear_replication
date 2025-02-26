import os
import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def xu_ly_du_lieu():
    st.write("Nội dung Xử lý dữ liệu...")
    # Thêm các chức năng xử lý dữ liệu ở đây

def phan_tich():
    st.write("Nội dung Phân tích...")
    # Thêm các chức năng phân tích dữ liệu ở đây

def demo():
    st.write("Nội dung Demo...")
    # Thêm các chức năng demo ở đây

def mlflow_section():
    st.write("Nội dung MLflow...")
    # Thêm các chức năng MLflow ở đây

def ly_thuyet_Decision_tree():
    st.write("Lý thuyết về Decision Tree...")
    # Thêm nội dung lý thuyết Decision Tree ở đây

def ly_thuyet_SVM():
    st.write("Lý thuyết về SVM...")
    # Thêm nội dung lý thuyết SVM ở đây

def data():
    st.write("Dữ liệu...")
    # Thêm nội dung hiển thị dữ liệu ở đây

def split_data():
    st.write("Chia dữ liệu...")
    # Thêm nội dung chia dữ liệu ở đây

def train():
    st.write("Huấn luyện mô hình...")
    # Thêm nội dung huấn luyện mô hình ở đây

def du_doan():
    st.write("Dự đoán...")
    # Thêm nội dung dự đoán ở đây

def main():
    st.title("MNIST Classification App")

    # Tạo sidebar để chọn dự án
    st.sidebar.title("Chọn dự án")
    project = st.sidebar.selectbox("Dự án", ("Khám phá dữ liệu", "Phân loại MNIST"))

    # Nếu người dùng chọn dự án "Khám phá dữ liệu"
    if project == "Khám phá dữ liệu":
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
            <div class="subtitle">Đề tài nghiên cứu</div>
            <hr>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader("Chọn các file dataset", accept_multiple_files=True)
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

            if error_files:
                st.warning("⚠️Các tệp không được chấp nhận: .Rhistory, body.txt, CrossTable, DocFile, file_example_XLS_100.xls. Vui lòng chỉ tải lên các tệp .csv hoặc .xlsx.⚠️")

        if datasets and not error_files:
            tab6, tab7, tab8, tab9 = st.tabs([
                "🗄️ Xử lý dữ liệu",
                "📊 Phân tích",
                "💡 Demo",
                "📝 MLflow"
            ])

            with tab6:
                xu_ly_du_lieu()

            with tab7:
                phan_tich()

            with tab8:
                demo()

            with tab9:
                mlflow_section()
        else:
            st.write("Vui lòng tải lên ít nhất một file dữ liệu để bắt đầu.")

    # Nếu người dùng chọn dự án "Phân loại MNIST"
    elif project == "Phân loại MNIST":
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📘 Lý thuyết Decision Tree",
            "📘 Lý thuyết SVM",
            "📘 Data",
            "⚙️ Huấn luyện",
            "🔢 Dự đoán"
        ])

        with tab1:
            ly_thuyet_Decision_tree()

        with tab2:
            ly_thuyet_SVM()

        with tab3:
            data()

        with tab4:
            split_data()
            train()

        with tab5:
            du_doan()

if __name__ == "__main__":
    main()
