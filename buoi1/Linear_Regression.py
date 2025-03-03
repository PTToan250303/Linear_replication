import streamlit as st
import pandas as pd
from scipy.stats import zscore
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
import mlflow
import io
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


import os
from mlflow.tracking import MlflowClient
from scipy.stats import zscore 
def mlflow_input():
    st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/PTToan250303/Linear_replication.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PTToan250303"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "5ca8caf353d564c358852da97c7487e64fc30a73"

    mlflow.set_experiment("Linear_replication")








def drop(df):
    st.subheader("🗑️ Xóa cột dữ liệu")
    
    if "df" not in st.session_state:
        st.session_state.df = df  # Lưu vào session_state nếu chưa có

    df = st.session_state.df
    columns_to_drop = st.multiselect("📌 Chọn cột muốn xóa:", df.columns.tolist())

    if st.button("🚀 Xóa cột đã chọn"):
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)  # Tạo bản sao thay vì inplace=True
            st.session_state.df = df  # Cập nhật session_state
            st.success(f"✅ Đã xóa cột: {', '.join(columns_to_drop)}")
            st.dataframe(df.head())
        else:
            st.warning("⚠️ Vui lòng chọn ít nhất một cột để xóa!")

    return df

def choose_label(df):
    st.subheader("🎯 Chọn cột dự đoán (label)")

    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    
    selected_label = st.selectbox("📌 Chọn cột dự đoán", df.columns, 
                                  index=df.columns.get_loc(st.session_state.target_column) if st.session_state.target_column else 0)

    X, y = df.drop(columns=[selected_label]), df[selected_label]  # Mặc định
    
    if st.button("✅ Xác nhận Label"):
        st.session_state.target_column = selected_label
        X, y = df.drop(columns=[selected_label]), df[selected_label]
        st.success(f"✅ Đã chọn cột: **{selected_label}**")
    
    return X, y

def train_test_size():
    if "df" not in st.session_state:
        st.error("❌ Dữ liệu chưa được tải lên!")
        st.stop()
    
    df = st.session_state.df  # Lấy dữ liệu từ session_stat
    X, y = choose_label(df)
    
    st.subheader("📊 Chia dữ liệu Train - Validation - Test")   
    
    test_size = st.slider("📌 Chọn % dữ liệu Test", 0, 100, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 100, 15)

    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    

    if st.button("✅ Xác nhận Chia"):
        # st.write("⏳ Đang chia dữ liệu...")

        stratify_option = y if y.nunique() > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        stratify_option = y_train_full if y_train_full.nunique() > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size / (100 - test_size),
            stratify=stratify_option, random_state=42
        )

        # Lưu vào session_state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.y = y
        st.session_state.X_train_shape = X_train.shape[0]
        st.session_state.X_val_shape = X_val.shape[0]
        st.session_state.X_test_shape = X_test.shape[0]
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })

        # **Log dữ liệu vào MLflow**
        

       
def xu_ly_gia_tri_thieu(df):
    if "df" not in st.session_state:
        st.session_state.df = df.copy()
    df = st.session_state.df

    # Tìm các cột có giá trị thiếu
    missing_cols = df.columns[df.isnull().any()].tolist()

    if not missing_cols:
        st.success("✅ Dữ liệu không có giá trị thiếu!")
        return df

    st.write("### 📌 Khi nào nên chọn phương pháp xử lý?")
    st.info("- **Xóa giá trị thiếu**: Nếu số lượng giá trị thiếu ít hoặc quá nhiều so với tổng dữ liệu.\n"
            "- **Thay thế bằng Mean (Trung bình)**: Nếu dữ liệu có phân phối chuẩn và không có quá nhiều outliers.\n"
            "- **Thay thế bằng Median (Trung vị)**: Nếu dữ liệu có nhiều phân phối lệch.\n"
            "- **Thay thế bằng Mode (Giá trị xuất hiện nhiều nhất)**: Nếu dữ liệu thuộc dạng phân loại (category).")

    selected_cols = st.multiselect("📌 Chọn cột chứa giá trị thiếu:", missing_cols)
    method = st.radio("🔧 Chọn phương pháp xử lý:", ["Xóa giá trị thiếu", "Thay thế bằng Mean", "Thay thế bằng Median", "Thay thế bằng Mode"])

    if st.button("🚀 Xử lý giá trị thiếu"):
        for col in selected_cols:
            if method == "Xóa giá trị thiếu":
                df = df.dropna(subset=[col])
            elif method == "Thay thế bằng Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "Thay thế bằng Median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "Thay thế bằng Mode":
                df[col] = df[col].fillna(df[col].mode()[0])
        
        st.session_state.df = df
        st.success(f"✅ Đã xử lý giá trị thiếu cho các cột đã chọn")
    
    st.dataframe(df.head())
    return df

def chuyen_doi_kieu_du_lieu(df):

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not categorical_cols:
        st.success("✅ Không có cột dạng chuỗi cần chuyển đổi!")
        return df
    st.write("Chuyển về kiểu dữ liệu số nguyên từ 1-n")
    selected_col = st.selectbox("📌 Cột cần chuyển đổi:", categorical_cols)
    unique_values = df[selected_col].unique()

    if "text_inputs" not in st.session_state:
        st.session_state.text_inputs = {}

    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []

    mapping_dict = {}
    input_values = []
    has_duplicate = False
    has_empty = False  # Kiểm tra nếu có ô trống

    st.write("### Các giá trị cần chuyển đổi:")
    for val in unique_values:
        st.write(f"- `{val}`")  # Hiển thị từng giá trị trên một dòng

    if len(unique_values) < 10:
        for val in unique_values:
            key = f"{selected_col}_{val}"
            if key not in st.session_state.text_inputs:
                st.session_state.text_inputs[key] = ""

            new_val = st.text_input(f"🔄 Nhập giá trị thay thế cho `{val}`:", 
                                    key=key, 
                                    value=st.session_state.text_inputs[key])

            st.session_state.text_inputs[key] = new_val
            input_values.append(new_val)
            mapping_dict[val] = new_val

        # Kiểm tra ô trống
        if "" in input_values:
            has_empty = True

        # Kiểm tra trùng lặp
        duplicate_values = [val for val in input_values if input_values.count(val) > 1 and val != ""]
        if duplicate_values:
            has_duplicate = True
            st.warning(f"⚠ Giá trị `{', '.join(set(duplicate_values))}` đã được sử dụng nhiều lần. Vui lòng chọn số khác!")

        # Nút bị mờ nếu có trùng hoặc chưa nhập đủ giá trị
        btn_disabled = has_duplicate or has_empty

        if st.button("🚀 Chuyển đổi dữ liệu", disabled=btn_disabled):
            column_info = {"column_name": selected_col, "mapping_dict": mapping_dict}
            st.session_state.mapping_dicts.append(column_info)

            df[selected_col] = df[selected_col].map(lambda x: mapping_dict.get(x, x))
            df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce')

            st.session_state.text_inputs.clear()
            st.session_state.df = df
            st.success(f"✅ Đã chuyển đổi cột `{selected_col}`")

    st.dataframe(df.head())
    return df









def chuan_hoa_du_lieu(df):
    # st.subheader("📊 Chuẩn hóa dữ liệu với StandardScaler")

    # Lọc tất cả các cột số
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Tìm các cột nhị phân (chỉ chứa 0 và 1)
    binary_cols = [col for col in numerical_cols if df[col].dropna().isin([0, 1]).all()]

    # Loại bỏ cột nhị phân khỏi danh sách cần chuẩn hóa
    cols_to_scale = list(set(numerical_cols) - set(binary_cols))

    if not cols_to_scale:
        st.success("✅ Không có thuộc tính dạng số cần chuẩn hóa!")
        return df

    if st.button("🚀 Thực hiện Chuẩn hóa"):
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Lưu vào session_state
        st.session_state.df = df

        st.success(f"✅ Đã chuẩn hóa xong")
        st.dataframe(df.head())

    return df

def hien_thi_ly_thuyet(df):

                # Kiểm tra lỗi dữ liệu
    st.subheader("🚨 Kiểm tra dữ liệu")
                # Kiểm tra giá trị thiếu
    missing_values = df.isnull().sum()

                # Kiểm tra dữ liệu trùng lặp
    duplicate_count = df.duplicated().sum()

                # Tạo báo cáo lỗi
    error_report = pd.DataFrame({
        'Giá trị thiếu': missing_values,
        'Dữ liệu trùng lặp': duplicate_count,
        'Tỉ lệ trùng lặp (%)': round((duplicate_count / df.shape[0]) * 100,2),
        'Kiểu dữ liệu': df.dtypes.astype(str)
    })

                # Hiển thị báo cáo lỗi
    st.table(error_report)          
   
    
    st.title("🔍 Tiền xử lý dữ liệu")

    # Hiển thị dữ liệu gốc
    
    st.header("⚙️ Các bước chính trong tiền xử lý dữ liệu")
    st.subheader("1️⃣ Loại bỏ các cột không cần thiết")


    df=drop(df)
    
    st.subheader("2️⃣ Xử lý giá trị thiếu")
    df=xu_ly_gia_tri_thieu(df)

    st.subheader("3️⃣ Chuyển đổi kiểu dữ liệu")

    df=chuyen_doi_kieu_du_lieu(df)
    
    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
 
    df=chuan_hoa_du_lieu(df)
    
def chia():
    st.subheader("Chia dữ liệu thành tập Train, Validation, và Test")
    st.write("""
    ### 📌 Chia tập dữ liệu
    Dữ liệu được chia thành ba phần để đảm bảo mô hình tổng quát tốt:
    - **Trian(%)**: để train mô hình.
    - **Val (%)**: để validation, dùng để điều chỉnh tham số.
    - **Test(%)**: để test, đánh giá hiệu suất thực tế.
    """)
    train_test_size()
    
    


def train_multiple_linear_regression(X_train, y_train, learning_rate=0.001, n_iterations=200):
    """Huấn luyện hồi quy tuyến tính bội bằng Gradient Descent."""
    
    # Chuyển đổi X_train, y_train sang NumPy array để tránh lỗi
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Kiểm tra NaN hoặc Inf
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị NaN!")
    if np.isinf(X_train).any() or np.isinf(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị vô cùng (Inf)!")

    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_train.shape
    #st.write(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1) vào X_train
    X_b = np.c_[np.ones((m, 1)), X_train]
    #st.write(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01  
    #st.write(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra xem gradients có NaN không
        # st.write(gradients)
        if np.isnan(gradients).any():
            raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    #st.success("✅ Huấn luyện hoàn tất!")
    #st.write(f"Trọng số cuối cùng: {w.flatten()}")
    return w
def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iterations=500):
    """Huấn luyện hồi quy đa thức **không có tương tác** bằng Gradient Descent."""

    # Chuyển dữ liệu sang NumPy array nếu là pandas DataFrame/Series
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Tạo đặc trưng đa thức **chỉ thêm bậc cao, không có tương tác**
    X_poly = np.hstack([X_train] + [X_train**d for d in range(2, degree + 1)])
    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = MinMaxScaler()
    X_poly = scaler.fit_transform(X_poly)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_poly.shape
    print(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1)
    X_b = np.c_[np.ones((m, 1)), X_poly]
    print(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01  
    print(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra nếu gradient có giá trị NaN
        if np.isnan(gradients).any():
            raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    print("✅ Huấn luyện hoàn tất!")
    print(f"Trọng số cuối cùng: {w.flatten()}")
    
    return w



# Hàm chọn mô hình
def chon_mo_hinh():
    st.subheader("🔍 Chọn mô hình hồi quy")
    
    model_type_V = st.radio("Chọn loại mô hình:", ["Multiple Linear Regression", "Polynomial Regression"])
    model_type = "linear" if model_type_V == "Multiple Linear Regression" else "polynomial"
    
    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5)
    learning_rate = st.slider("Chọn tốc độ học (learning rate):", 
                          min_value=1e-6, max_value=0.1, value=0.01, step=1e-6, format="%.6f")

    degree = 2
    if model_type == "polynomial":
        degree = st.slider("Chọn bậc đa thức:", min_value=2, max_value=5, value=2)

    fold_mse = []
    scaler = MinMaxScaler()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    if "X_train" not in st.session_state or st.session_state.X_train is None:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi huấn luyện mô hình!")
        return None, None, None

    X_train, X_test = st.session_state.X_train, st.session_state.X_test
    y_train, y_test = st.session_state.y_train, st.session_state.y_test
    
    mlflow_input()
    
    # Lưu vào session_state để không bị mất khi cập nhật UI
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")  # Tên run cho MLflow
    st.session_state["run_name"] = run_name if run_name else "default_run"
    
    if st.button("Huấn luyện mô hình"):
        # 🎯 **Tích hợp MLflow**
        

        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            df = st.session_state.df
            mlflow.log_param("dataset_shape", df.shape)
            mlflow.log_param("target_column", st.session_state.y.name)
            mlflow.log_param("test_size", st.session_state.X_test_shape)
            mlflow.log_param("validation_size", st.session_state.X_val_shape)
            mlflow.log_param("train_size", st.session_state.X_train_shape)

            # Lưu dataset tạm thời
            dataset_path = "dataset.csv"
            df.to_csv(dataset_path, index=False)

            # Log dataset lên MLflow
            mlflow.log_artifact(dataset_path)


            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_folds", n_folds)
            mlflow.log_param("learning_rate", learning_rate)
            if model_type == "polynomial":
                mlflow.log_param("degree", degree)

            for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
                X_train_fold, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                y_train_fold, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                if model_type == "linear":
                    w = train_multiple_linear_regression(X_train_fold, y_train_fold, learning_rate=learning_rate)
                    w = np.array(w).reshape(-1, 1)
                    X_valid_b = np.c_[np.ones((len(X_valid), 1)), X_valid.to_numpy()]
                    y_valid_pred = X_valid_b.dot(w)
                else:  
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    w = train_polynomial_regression(X_train_fold, y_train_fold, degree, learning_rate=learning_rate)
                    w = np.array(w).reshape(-1, 1)
                    X_valid_scaled = scaler.transform(X_valid.to_numpy())
                    X_valid_poly = np.hstack([X_valid_scaled] + [X_valid_scaled**d for d in range(2, degree + 1)])
                    X_valid_b = np.c_[np.ones((len(X_valid_poly), 1)), X_valid_poly]
                    y_valid_pred = X_valid_b.dot(w)

                mse = mean_squared_error(y_valid, y_valid_pred)
                fold_mse.append(mse)
                mlflow.log_metric(f"mse_fold_{fold+1}", mse)
                print(f"📌 Fold {fold + 1} - MSE: {mse:.4f}")

            avg_mse = np.mean(fold_mse)

            if model_type == "linear":
                final_w = train_multiple_linear_regression(X_train, y_train, learning_rate=learning_rate)
                st.session_state['linear_model'] = final_w
                X_test_b = np.c_[np.ones((len(X_test), 1)), X_test.to_numpy()]
                y_test_pred = X_test_b.dot(final_w)
            else:
                X_train_scaled = scaler.fit_transform(X_train)
                final_w = train_polynomial_regression(X_train_scaled, y_train, degree, learning_rate=learning_rate)
                st.session_state['polynomial_model'] = final_w
                X_test_scaled = scaler.transform(X_test.to_numpy())
                X_test_poly = np.hstack([X_test_scaled] + [X_test_scaled**d for d in range(2, degree + 1)])
                X_test_b = np.c_[np.ones((len(X_test_poly), 1)), X_test_poly]
                y_test_pred = X_test_b.dot(final_w)

            test_mse = mean_squared_error(y_test, y_test_pred)

            # 📌 **Log các giá trị vào MLflow**
            mlflow.log_metric("avg_mse", avg_mse)
            mlflow.log_metric("test_mse", test_mse)

            # Kết thúc run
            mlflow.end_run()
            
            st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")
            st.success(f"MSE trên tập test: {test_mse:.4f}")
            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
            st.markdown(f"### 🔗 [Truy cập MLflow DAGsHub]({st.session_state['mlflow_url']})")

        return final_w, avg_mse, scaler

    return None, None, None

def test():
    # Kiểm tra xem mô hình đã được lưu trong session_state chưa
    model_type = st.selectbox("Chọn mô hình:", ["linear", "polynomial"])

    if model_type == "linear" and "linear_model" in st.session_state:
        model = st.session_state["linear_model"]
    elif model_type == "polynomial" and "polynomial_model" in st.session_state:
        model = st.session_state["polynomial_model"]
    else:
        st.warning("Mô hình chưa được huấn luyện.")
        return

    # Nhập các giá trị cho các cột của X_train
    X_train = st.session_state.X_train
    
    st.write(X_train.head()) 
    
    # Đảm bảo bạn dùng session_state
    num_columns = len(X_train.columns)
    column_names = X_train.columns.tolist()

    st.write(f"Nhập các giá trị cho {num_columns} cột của X_train:")

    # Tạo các trường nhập liệu cho từng cột
    X_train_input = []
    binary_columns = [] 
    # Kiểm tra nếu có dữ liệu mapping_dicts trong session_state
    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []

    # Duyệt qua các cột và kiểm tra nếu có thông tin chuyển đổi
    for i, column_name in enumerate(column_names):
        # Kiểm tra xem cột có nằm trong mapping_dicts không
        mapping_dict = None
        for column_info in st.session_state.mapping_dicts:
            if column_info["column_name"] == column_name:
                mapping_dict = column_info["mapping_dict"]
                break

        if mapping_dict:  # Nếu có mapping_dict, hiển thị dropdown với các giá trị thay thế
            value = st.selectbox(f"Giá trị cột {column_name}", options=list(mapping_dict.keys()), key=f"column_{i}")
            value = int(mapping_dict[value])
        else:  # Nếu không có mapping_dict, yêu cầu người dùng nhập số
            value = st.number_input(f"Giá trị cột {column_name}", key=f"column_{i}")
        
        X_train_input.append(value)
    
    # Chuyển đổi list thành array
    X_train_input = np.array(X_train_input).reshape(1, -1)

    # Sao chép X_train_input để thay đổi giá trị không làm ảnh hưởng đến dữ liệu gốc
    X_train_input_final = X_train_input.copy()  
    scaler = MinMaxScaler()

    # Tạo mảng chỉ số của các phần tử khác 0 và 1
    for i in range(X_train_input.shape[1]):
        if X_train_input[0, i] != 0 and X_train_input[0, i] != 1:  # Nếu giá trị không phải 0 hoặc 1
            # Chuẩn hóa giá trị
            X_train_input_final[0, i] = scaler.fit_transform(X_train_input[:, i].reshape(-1, 1)).flatten()
        
    st.write("Dữ liệu sau khi xử lý:")
    st.write(X_train_input_final)

    if st.button("Dự đoán"):
        # Thêm cột 1 cho intercept (nếu cần)
        X_input_b = np.c_[np.ones((X_train_input_final.shape[0], 1)), X_train_input_final]
        
        # Dự đoán với mô hình đã lưu
        y_pred = X_input_b.dot(model)  # Dự đoán với mô hình đã lưu
        
        # Hiển thị kết quả dự đoán
        if y_pred >= 0.5:
            st.write("Dự đoán sống 🎈")        
        else:
            st.write("Dự đoán chết 💀")

def data(df):
    """Hiển thị dữ liệu đã tải lên"""
    if df is not None:
        st.success("📂 File đã được tải lên thành công!")
        hien_thi_ly_thuyet(df)
    else:
        st.error("❌ Không có dữ liệu để hiển thị.")
            
import streamlit as st
import mlflow
import os

import streamlit as st
import mlflow
import os
import pandas as pd
from datetime import datetime
def show_experiment_selector():
    st.title("📊 MLflow Experiments - DAGsHub")

    # Kết nối với DAGsHub MLflow Tracking
    
    # Lấy danh sách tất cả experiments
    experiment_name = "Linear_replication"
    
    # Tìm experiment theo tên
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách runs trong experiment
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")

    # Lấy danh sách run_name từ params
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_params = mlflow.get_run(run_id).data.params
        run_name = run_params.get("run_name", f"Run {run_id[:8]}")  # Nếu không có run_name thì lấy run_id
        run_info.append((run_name, run_id))

    # Tạo dictionary để map run_name -> run_id
    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())

    # Chọn run theo run_name
    selected_run_name = st.selectbox("🔍 Chọn một run:", run_names)
    selected_run_id = run_name_to_id[selected_run_name]

    # Hiển thị thông tin chi tiết của run được chọn
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        start_time_ms = selected_run.info.start_time  # Thời gian lưu dưới dạng milliseconds

# Chuyển sang định dạng ngày giờ dễ đọc
        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        # Kiểm tra và hiển thị dataset artifact
        dataset_path = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/dataset.csv"
        st.write("### 📂 Dataset:")
        st.write(f"📥 [Tải dataset]({dataset_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")
def chon():
    try:
                
        final_w, avg_mse, scaler = chon_mo_hinh()
    except Exception as e:
        st.error(f"Lỗi xảy ra: {e}")
def Classification():
    # Định dạng tiêu đề
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
        <div class="title">Khái phá dữ liệu</div>
        <div class="subtitle">Linear Regression</div>
        <hr>
    """, unsafe_allow_html=True)

    # Cho phép người dùng tải một file duy nhất
    uploaded_file = st.file_uploader("📥 Chọn một file dataset", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.write("Định dạng tệp CSV hợp lệ.")
            else:
                st.error("❌ Định dạng tệp không hợp lệ. Vui lòng tải lại tệp .csv")
                return  # Dừng chương trình nếu tải sai file
        except Exception as e:
            st.error(f"⚠️ Lỗi khi đọc tệp: {e}")
            return

        st.success(f"✅ Đã tải lên: {uploaded_file.name}")
        st.write(df)  # Hiển thị toàn bộ dataset

        # Chỉ hiển thị thanh điều hướng khi có file hợp lệ
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📘 Lý thuyết", 
            "📊 Xử lý dữ liệu",
            "⚙️ Huấn luyện", 
            "💡 Demo",
            "📝 MLflow"
        ])

        with tab1:
            st.write("Lý thuyết ")
        with tab2:
            data(df)
        with tab3:
            chia()
            chon()
        with tab4:
            test()
        with tab5:
            show_experiment_selector()

if __name__ == "__main__":
    Classification()
