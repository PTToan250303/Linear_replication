import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import os
import random
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from datetime import datetime

# === Cấu hình MLflow ===
DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

if "mlflow_url" not in st.session_state:
    st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
    mlflow.set_experiment("Classification") 
st.set_page_config(page_title="MNIST Clustering App", layout="wide")


# === Load dữ liệu MNIST ===
def load_mnist_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.astype(np.float64) / 255.0  # Đảm bảo dữ liệu có dtype phù hợp
    return X, y.astype(int)

# === Chia dữ liệu ===
def split_data():
    st.header("📌 Chia dữ liệu MNIST")
    X, y = load_mnist_data()
    
    test_size = st.slider("Chọn tỷ lệ Test (%)", 10, 50, 20) / 100
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.session_state["X_train"], st.session_state["X_test"] = X_train, X_test
    st.session_state["y_train"], st.session_state["y_test"] = y_train, y_test
    
    st.success("✅ Dữ liệu đã được chia thành công!")

# === Huấn luyện mô hình ===
def train():
    st.header("⚙️ Huấn luyện mô hình")
    
    if "X_train" not in st.session_state:
        st.error("⚠️ Hãy chia dữ liệu trước khi huấn luyện!")
        return
    
    X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]
    
    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"], key="train_model_select")
    
    if model_choice == "K-Means":
        k = st.slider("Chọn số cụm (K):", 2, 20, 10)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    else:
        eps = st.slider("Bán kính lân cận (eps):", 0.1, 10.0, 0.5)
        min_samples = st.slider("Số điểm tối thiểu để tạo cụm:", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    
    if st.button("🚀 Huấn luyện"):
        with mlflow.start_run():
            model.fit(X_train.astype(np.float64))  # Đảm bảo dtype phù hợp
            mlflow.sklearn.log_model(model, "model")
            joblib.dump(model, f"{model_choice.lower()}_model.joblib")
            st.success("✅ Huấn luyện thành công!")

# === Vẽ số trên Canvas để phân cụm ===
def draw_and_predict():
    st.header("✍️ Vẽ số để phân cụm")
    
    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"], key="predict_model_select")
    model_filename = f"{model_choice.lower()}_model.joblib"
    
    if not os.path.exists(model_filename):
        st.error("⚠️ Mô hình chưa được huấn luyện!")
        return
    
    model = joblib.load(model_filename)
    
    if "canvas_key" not in st.session_state:
        st.session_state["canvas_key"] = str(random.randint(0, 1000000))
    
    if st.button("🔄 Tải lại"):
        st.session_state["canvas_key"] = str(random.randint(0, 1000000))
        st.rerun()
    
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state["canvas_key"],
    )
    
    if st.button("📊 Dự đoán cụm"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
            img = img.resize((28, 28)).convert("L")
            img = np.array(img).reshape(1, -1) / 255.0  # Chuẩn hóa
            img = img.astype(np.float64)  # Đảm bảo dtype
            cluster = model.predict(img)[0] if isinstance(model, KMeans) else "Không xác định"
            st.subheader(f"🔢 Cụm dự đoán: {cluster}")
        else:
            st.warning("⚠ Vẽ số trước khi dự đoán!")
def show_experiment_selector():
    st.title("📊 MLflow")
    
    mlflow.set_tracking_uri("https://dagshub.com/PTToan250303/Linear_replication.mlflow")
    
    experiment_name = "Clustering"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")
    
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_params = mlflow.get_run(run_id).data.params
        run_name = run_params.get("run_name", f"Run {run_id[:8]}")
        run_info.append((run_name, run_id))
    
    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())
    
    selected_run_name = st.selectbox("🔍 Chọn một run:", run_names)
    selected_run_id = run_name_to_id[selected_run_name]

    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        
        start_time_ms = selected_run.info.start_time
        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"
        
        st.write(f"**Thời gian chạy:** {start_time}")

        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        model_type = params.get("model", "Unknown")
        if model_type == "K-Means":
            st.write(f"🔹 **Mô hình:** K-Means")
            st.write(f"🔢 **Số cụm (K):** {params.get('n_clusters', 'N/A')}")
            st.write(f"🎯 **Độ chính xác:** {metrics.get('accuracy', 'N/A')}")
        elif model_type == "DBSCAN":
            st.write(f"🛠️ **Mô hình:** DBSCAN")
            st.write(f"📏 **eps:** {params.get('eps', 'N/A')}")
            st.write(f"👥 **Min Samples:** {params.get('min_samples', 'N/A')}")
            st.write(f"🔍 **Số cụm tìm thấy:** {metrics.get('n_clusters_found', 'N/A')}")
            st.write(f"🚨 **Tỉ lệ nhiễu:** {metrics.get('noise_ratio', 'N/A')}")

        model_artifact_path = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/{model_type.lower()}_model"
        st.write("### 📂 Model Artifact:")
        st.write(f"📥 [Tải mô hình]({model_artifact_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")

# === Giao diện Streamlit ===
def main():
    st.title("🖊️ MNIST Clustering App")

    tab1, tab2, tab3, tab4 = st.tabs(["📘 Dữ liệu", "⚙️ Huấn luyện", "🔢 Dự đoán", "🔥 MLflow"])

    with tab1:
        split_data()
    with tab2:
        train()
    with tab3:
        draw_and_predict()
    with tab4:
        show_experiment_selector()
        st.write(f"🔗 [Truy cập MLflow]({st.session_state['mlflow_url']})")

if __name__ == "__main__":
    main()