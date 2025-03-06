import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import joblib
import mlflow
import os
import random
import cv2

# === Cấu hình MLflow ===
DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
mlflow.set_experiment("Classification")

st.set_page_config(page_title="MNIST Classification App", layout="wide")

# === Load dữ liệu MNIST ===
def load_mnist_data():
    X = np.load("X.npy")
    y = np.load("y.npy")
    return X, y

# === Chia dữ liệu ===
def split_data():
    st.header("📌 Chia dữ liệu MNIST")
    X, y = load_mnist_data()
    
    st.write(f"Tổng số mẫu: {len(y)}")
    
    test_size = st.slider("Chọn tỷ lệ Test (%)", 10, 50, 20) / 100
    val_size = st.slider("Chọn tỷ lệ Validation (%)", 0, 50, 15) / 100
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, random_state=42)
    
    st.session_state["X_train"], st.session_state["X_val"], st.session_state["X_test"] = X_train, X_val, X_test
    st.session_state["y_train"], st.session_state["y_val"], st.session_state["y_test"] = y_train, y_val, y_test
    
    st.write(f"📊 Kích thước tập Train: {len(y_train)} mẫu")
    st.write(f"📊 Kích thước tập Validation: {len(y_val)} mẫu")
    st.write(f"📊 Kích thước tập Test: {len(y_test)} mẫu")
    
    st.success("✅ Dữ liệu đã được chia thành công!")

# === Tiền xử lý ảnh đầu vào ===
def preprocess_image(img):
    img = np.array(img)
    
    # Đảm bảo ảnh có đúng số kênh màu
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize ảnh về 28x28
    img = cv2.resize(img, (28, 28))
    
    # Đảm bảo chữ trắng, nền đen
    if np.mean(img) > 127:  # Nếu nền tối hơn chữ, đảo ngược màu
        img = cv2.bitwise_not(img)
    
    # Chuẩn hóa về [0,1]
    img = img.astype(np.float32) / 255.0
    
    # Reshape để đưa vào mô hình
    img = img.reshape(1, -1)
    
    return img
# === Huấn luyện mô hình ===
def train():
    if "X_train" not in st.session_state:
        st.error("⚠️ Hãy chia dữ liệu trước khi huấn luyện!")
        return
    
    X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]
    X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]
    
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0
    
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"], key="train_model_select")
    
    if model_choice == "Decision Tree":
        max_depth = st.slider("max_depth", 1, 20, 5, key="tree_depth")
        model = DecisionTreeClassifier(max_depth=max_depth)
        model_filename = "models/decision_tree.joblib"
    else:
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, key="svm_c")
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], key="svm_kernel")
        model = SVC(C=C, kernel=kernel)
        model_filename = "models/svm.joblib"
    
    n_folds = st.slider("Chọn số folds (Cross-Validation):", 2, 10, 5, key="cv_folds")
    
    if st.button("Huấn luyện mô hình"):
        with mlflow.start_run():
            mlflow.log_param("model", model_choice)
            mlflow.log_param("cross_validation_folds", n_folds)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds)
            mean_cv_score = cv_scores.mean()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("cv_accuracy", mean_cv_score)
            mlflow.sklearn.log_model(model, "model")
            
            joblib.dump(model, model_filename)
            st.success(f"📊 Độ chính xác trên tập test: {test_acc:.4f}")
def get_run_name():
    return st.text_input("🔖 Nhập tên Run:", "Default_Run")
# === Dự đoán số viết tay ===
def predict_digit():
    st.header("✍️ Vẽ số hoặc tải ảnh để dự đoán")
    
    model_option = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"], key="predict_model_select")
    model_filename = f"models/{'decision_tree' if model_option == 'Decision Tree' else 'svm'}.joblib"
    
    if not os.path.exists(model_filename):
        st.error(f"⚠️ Mô hình {model_option} chưa được huấn luyện. Hãy huấn luyện trước!")
        return
    
    model = joblib.load(model_filename)
    uploaded_file = st.file_uploader("📤 Tải lên ảnh số viết tay", type=["png", "jpg", "jpeg"])
    
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))
    
    if st.button("🔄 Tải lại nếu không thấy canvas"):
        st.session_state.key_value = str(random.randint(0, 1000000))
        st.rerun()

    st.write("🖌️ Vẽ số vào bảng dưới:")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state.key_value,
    )
    
    if st.button("📊 Dự đoán số"):
        if uploaded_file:
            img = Image.open(uploaded_file).convert("L")  # Chuyển ảnh tải lên sang grayscale
        elif canvas_result.image_data is not None:
            img_array = canvas_result.image_data
            img = Image.fromarray(img_array[:, :, 0].astype(np.uint8))
        else:
            st.warning("⚠ Vui lòng tải lên ảnh hoặc vẽ số trên canvas!")
            return
        
        # 🔹 Hiển thị ảnh trước khi tiền xử lý
        st.image(img, caption="Ảnh gốc", width=150)
        
        # 🔹 Tiền xử lý ảnh
        img = preprocess_image(img)
        
        # 🔹 Hiển thị ảnh sau khi tiền xử lý
        st.image(img.reshape(28, 28), caption="Ảnh sau tiền xử lý", width=150)
        
        # 🔹 Đưa vào mô hình để dự đoán
        prediction = model.predict(img)

        st.subheader(f"🔢 Dự đoán: {prediction[0]}")


def show_experiment_selector():
    st.title("📊 MLflow Experiments - DAGsHub")
    experiment_name = "Classification"
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
    
    if not run_info:
        st.warning("⚠ Không có run hợp lệ trong experiment này.")
        return
    
    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())
    
    selected_run_name = st.selectbox("🔍 Chọn một run:", run_names) if run_names else None
    
    if not selected_run_name or selected_run_name not in run_name_to_id:
        st.warning("⚠ Vui lòng chọn một run hợp lệ!")
        return
    
    selected_run_id = run_name_to_id[selected_run_name]
    selected_run = mlflow.get_run(selected_run_id)
    
    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        start_time_ms = selected_run.info.start_time
        start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
        st.write(f"**Thời gian chạy:** {start_time}")
        st.write("### ⚙️ Parameters:")
        st.json(selected_run.data.params)
        st.write("### 📊 Metrics:")
        st.json(selected_run.data.metrics)
        mlflow_ui_url = f"{DAGSHUB_MLFLOW_URI}/experiments/{selected_experiment.experiment_id}/runs/{selected_run_id}"
        st.markdown(f"🔗 [Truy cập MLflow trên DAGsHub]({mlflow_ui_url})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")
def Classification():
  
    if "mlflow_initialized" not in st.session_state:   
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
        st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI

        st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"
        mlflow.set_experiment("Classification")   
    st.markdown("""
        <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #4682B4;
            margin-top: 50px;
        }
        .subtitle {
            font-size: 24px;
            text-align: center;
            color: #4A4A4A;
        }
        hr {
            border: 1px solid #ddd;
        }
        </style>
        <div class="title">MNIST Classification App</div>
        <hr>
    """, unsafe_allow_html=True)    
    
    #st.session_state.clear()
    ### **Phần 1: Hiển thị dữ liệu MNIST**
    
    ### **Phần 2: Trình bày lý thuyết về Decision Tree & SVM*
    
    # 1️⃣ Phần giới thiệu
    
    # === Sidebar để chọn trang ==
    # === Tạo Tabs ===
    tab1, tab2, tab3, tab4,tab5 ,tab6= st.tabs(["📘 Lý thuyết Decision Tree", "📘 Lý thuyết SVM", "📘 Data" ,"⚙️ Huấn luyện", "🔢 Dự đoán","🔥Mlflow"])
    
    with tab1:
          st.write("Lý thuyết ")

    with tab2:
          st.write("Lý thuyết ")
    with tab3:
        split_data()
        
    with tab4:
       # plot_tree_metrics()
        
        
        
        train()
        run_name = get_run_name()
    with tab5:
        
        predict_digit() 
    with tab6:
        
        show_experiment_selector()  

if __name__ == "__main__":
    Classification()