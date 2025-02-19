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
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import os

# Đọc dữ liệu
df = pd.read_csv('titanic.csv')

# Hiển thị bảng dữ liệu gốc, số hàng và số cột, tổng số giá trị thiếu trong từng cột
st.title("Original Data")
st.write("**Bảng dữ liệu gốc:**")
st.dataframe(df)

st.write(f"**Số hàng và số cột:** {df.shape[0]} x {df.shape[1]}")
st.write("**Tổng số giá trị thiếu trong từng cột (dữ liệu gốc):**")
missing_values_original = df.isnull().sum()
st.write(missing_values_original)

# Tiền xử lý dữ liệu
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

st.write("**Ánh xạ các giá trị trong cột Embarked thành các giá trị số: 'S'=0, 'C'=1, và 'Q'=2.**")
st.write("**Sau đó điền giá trị thiếu của cột Embarked bằng giá trị phổ biến nhất (mode).**")

st.write("**Cột Cabin vì dữ liệu thiếu quá nhiều nên chúng ta sẽ loại bỏ nó luôn**")
st.write("**Còn cột Age thì điền giá trị thiếu bằng giá trị trung bình(mean)**")

columns_to_convert = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Sex']
for col in columns_to_convert:
    df[col] = df[col].astype('float64')

# Hiển thị bảng dữ liệu sau khi tiền xử lý, số hàng và số cột, tổng số giá trị thiếu trong từng cột
st.title("Processed Data")
st.write("**Bảng dữ liệu sau khi tiền xử lý:**")
st.dataframe(df)

st.write(f"**Số hàng và số cột:** {df.shape[0]} x {df.shape[1]}")
st.write("**Tổng số giá trị thiếu trong từng cột (sau khi tiền xử lý):**")
missing_values_processed = df.isnull().sum()
st.write(missing_values_processed)

# Chuẩn hóa tất cả các cột trừ 'Survived'
scaler = StandardScaler()
columns_to_scale = df.columns.difference(['Name', 'Ticket', 'Survived'])
df_scaled = df.copy()
df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Hiển thị bảng dữ liệu sau khi chuẩn hóa
st.title("Normalized Data")
st.write("**Bảng dữ liệu sau khi chuẩn hóa:**")
st.dataframe(df_scaled)

# Chia dữ liệu thành tập huấn luyện, tập xác thực và tập kiểm thử
train, temp = train_test_split(df_scaled, test_size=0.3, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

X_train = train.drop(["Survived", "Name", "Ticket"], axis=1)
X_valid = valid.drop(["Survived", "Name", "Ticket"], axis=1)
X_test = test.drop(["Survived", "Name", "Ticket"], axis=1)
Y_train = train["Survived"]
Y_valid = valid["Survived"]
Y_test = test["Survived"]

# Kiểm tra xem cột có tồn tại trước khi chuyển đổi kiểu dữ liệu
for col in columns_to_convert:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('float64')
    if col in X_valid.columns:
        X_valid[col] = X_valid[col].astype('float64')

# Bước 1: Cross Validation trên tập huấn luyện
model = LogisticRegression()
cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
st.write("### Cross Validation Results (Training Data)")
st.write(f"**Accuracy Scores for each fold:** {cv_scores}")
st.write(f"**Mean Accuracy:** {cv_scores.mean()}")
st.write(f"**Standard Deviation:** {cv_scores.std()}")

# Bước 2: Huấn luyện mô hình trên toàn bộ tập dữ liệu huấn luyện
model.fit(X_train, Y_train)

# Bước 3: Đánh giá mô hình trên tập kiểm thử
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)

with mlflow.start_run() as run:
    run_id = run.info.run_id

    mlflow.log_param("Training Size", len(X_train))
    mlflow.log_param("Validation Size", len(X_valid))
    mlflow.log_param("Test Size", len(X_test))
    mlflow.log_param("random_state", 42)

    # Ghi lại dữ liệu chia tách vào file CSV
    train_file = f"train_data_{run_id}.csv"
    valid_file = f"valid_data_{run_id}.csv"
    test_file = f"test_data_{run_id}.csv"

    X_train.to_csv(train_file, index=False)
    X_valid.to_csv(valid_file, index=False)
    X_test.to_csv(test_file, index=False)

    mlflow.log_artifact(train_file)
    mlflow.log_artifact(valid_file)
    mlflow.log_artifact(test_file)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("accuracy", accuracy)

    input_example = X_train.iloc[[0]]
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)

    # Ứng dụng Streamlit
    st.title("Data Splitting Visualization")
    st.write("**Số lượng mẫu trong từng tập dữ liệu**")

    split_info = {
        "Train": len(X_train),
        "Validation": len(X_valid),
        "Test": len(X_test)
    }
    df_split = pd.DataFrame(list(split_info.items()), columns=["Dataset", "Size"])
    st.table(df_split)

    fig = px.bar(df_split, x="Dataset", y="Size", title="Data Split Overview", color="Dataset")
    st.plotly_chart(fig)

    st.write("**Mô tả:** Biểu đồ trên hiển thị số lượng mẫu trong Training, Validation và Test Set sau khi chia dữ liệu")

    st.write("### Model Training and Evaluation")
    st.write(f"**Mean Squared Error (MSE):** {mse}")
    st.write(f"**Accuracy:** {accuracy}")

    st.write("**Mô tả:** Chúng tôi đã huấn luyện một mô hình Logistic Regression và đánh giá nó trên tập kiểm thử. Độ chính xác và Mean Squared Error của mô hình được hiển thị ở trên.")

# Mô hình Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_valid_poly = poly.transform(X_valid)
X_test_poly = poly.transform(X_test)

poly_reg_model = make_pipeline(PolynomialFeatures(2), LogisticRegression())
poly_reg_model.fit(X_train_poly, Y_train)

st.write("### Dự Đoán Demo")

# Nhập dữ liệu từ người dùng
user_input = {
    'PassengerId': st.number_input('PassengerId', value=1.0),
    'Pclass': float(st.selectbox('Pclass', [1, 2, 3])),
    'Sex': float(st.selectbox('Sex', [0, 1])),
    'Age': st.number_input('Age', value=df['Age'].mean()),
    'SibSp': st.number_input('SibSp', value=0.0),
    'Parch': st.number_input('Parch', value=0.0),
    'Fare': st.number_input('Fare', value=df['Fare'].mean()),
    'Embarked': float(st.selectbox('Embarked', [0, 1, 2]))
}
# Chuyển dữ liệu người dùng thành DataFrame
user_data_df = pd.DataFrame([user_input])

# Chuẩn hóa dữ liệu đầu vào (trừ các cột không được chuẩn hóa)
user_data_df[columns_to_scale] = scaler.transform(user_data_df[columns_to_scale])

# Biến đổi thành Polynomial Features
user_data_poly = poly.transform(user_data_df)

# Dự đoán
prediction = poly_reg_model.predict(user_data_poly)

# Hiển thị kết quả dự đoán
st.write(f"**Dự đoán Survived (Polynomial Regression): {int(np.round(prediction[0]))}**")

# Tìm mẫu trong tập test gần giống nhất với dữ liệu người dùng
def find_closest_sample(user_data, X_test):
    min_distance = float('inf')
    closest_index = -1
    for i in range(len(X_test)):
        dist = euclidean(user_data, X_test.iloc[i])
        if dist < min_distance:
            min_distance = dist
            closest_index = i
    return closest_index

# Tìm mẫu gần nhất trong tập test
closest_index = find_closest_sample(user_data_df.iloc[0], X_test)
closest_sample = X_test.iloc[closest_index]
closest_sample_poly = poly.transform([closest_sample])

# Dự đoán cho mẫu gần nhất trong tập test
test_prediction = poly_reg_model.predict(closest_sample_poly)[0]
actual_label = Y_test.iloc[closest_index]  # Nhãn thực tế của mẫu gần nhất

# Kiểm tra xem dự đoán của người dùng có trùng với nhãn thực tế hay không
correct_prediction = (int(np.round(prediction[0])) == actual_label)

# Hiển thị kết quả kiểm tra
st.write(f"**Giá trị thực tế từ tập test: {actual_label}**")
st.write(f"**Kết quả dự đoán {'ĐÚNG' if correct_prediction else 'SAI'}**")

import os

# Đường dẫn đến thư mục `0` của MLflow
mlflow_folder = 'mlruns/0/'

# Hàm liệt kê các file trong thư mục
def list_files_in_folder(folder_path, keyword):
    files = []
    for root, dirs, file_names in os.walk(folder_path):
        for file_name in file_names:
            if keyword in file_name:
                files.append(os.path.join(root, file_name))
    return sorted(files, key=os.path.getmtime, reverse=True)

# Liệt kê các file train, test và validation trong thư mục `0`
train_files = list_files_in_folder(mlflow_folder, 'train')
test_files = list_files_in_folder(mlflow_folder, 'test')
val_files = list_files_in_folder(mlflow_folder, 'val')

# Hiển thị các file trên Streamlit
st.title('Files in MLflow Folder')

tab1, tab2, tab3 = st.tabs(["Train Files", "Test Files", "Validation Files"])

with tab1:
    st.write("### Train Files")
    selected_train_file = st.selectbox('Chọn file train để xem:', train_files)
    if selected_train_file:
        st.write(f'**Nội dung của file {selected_train_file}:**')
        if selected_train_file.endswith('.csv'):
            df_train = pd.read_csv(selected_train_file)
            st.dataframe(df_train)
        else:
            with open(selected_train_file, 'r') as file:
                content_train = file.read()
                st.text(content_train)

with tab2:
    st.write("### Test Files")
    selected_test_file = st.selectbox('Chọn file test để xem:', test_files)
    if selected_test_file:
        st.write(f'**Nội dung của file {selected_test_file}:**')
        if selected_test_file.endswith('.csv'):
            df_test = pd.read_csv(selected_test_file)
            st.dataframe(df_test)
        else:
            with open(selected_test_file, 'r') as file:
                content_test = file.read()
                st.text(content_test)

with tab3:
    st.write("### Validation Files")
    selected_val_file = st.selectbox('Chọn file validation để xem:', val_files)
    if selected_val_file:
        st.write(f'**Nội dung của file {selected_val_file}:**')
        if selected_val_file.endswith('.csv'):
            df_val = pd.read_csv(selected_val_file)
            st.dataframe(df_val)
        else:
            with open(selected_val_file, 'r') as file:
                content_val = file.read()
                st.text(content_val)
