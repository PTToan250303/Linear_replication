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
        def ly_thuyet_Random_Forest():
            st.title("Random Forest")

            st.write("""
            Random forest là thuật toán supervised learning, có thể giải quyết cả bài toán regression và classification.
                     
            ### 1. Giới thiệu
            Random là ngẫu nhiên, Forest là rừng, nên ở thuật toán Random Forest mình sẽ xây dựng nhiều cây quyết định bằng thuật toán Decision Tree, tuy nhiên mỗi cây quyết định sẽ khác nhau (có yếu tố random). Sau đó kết quả dự đoán được tổng hợp từ các cây quyết định.
            
            Ở bước huấn luyện thì mình sẽ xây dựng nhiều cây quyết định, các cây quyết định có thể khác nhau (phần sau mình sẽ nói mỗi cây được xây dựng như thế nào).
            """)
            st.image("imageB1/random_forest.png", use_container_width=True)
            st.write("""
            Sau đó ở bước dự đoán, với một dữ liệu mới, thì ở mỗi cây quyết định mình sẽ đi từ trên xuống theo các node điều kiện để được các dự đoán, sau đó kết quả cuối cùng được tổng hợp từ kết quả của các cây quyết định.            
            """)

            st.image("imageB1/random_forest_predict.png", use_container_width=True)
            st.write("""
            Ví dụ như trên, thuật toán Random Forest có 6 cây quyết định, 5 cây dự đoán 1 và 1 cây dự đoán 0, do đó mình sẽ vote là cho ra dự đoán cuối cùng là 1.
    
            ### 2. Xây dựng thuật toán Random Forest
            
            Giả sử bộ dữ liệu của mình có n dữ liệu (sample) và mỗi dữ liệu có d thuộc tính (feature).
            
            Đề xây dựng mỗi cây quyết định minh họa sau:
            """)

            st.markdown("""
            <div style="margin-left: 30px; line-height: 1.5;"> 
            Lấy ngẫu nhiên n dữ liệu từ bộ dữ liệu với kĩ thuật Bootstrapping, hay còn gọi là random sampling with replacement. Tức khi mình sample được 1 dữ liệu thì mình không bỏ dữ liệu đấy ra mà vẫn giữ lại trong tập dữ liệu ban đầu, rồi tiếp tục sample cho tới khi sample đủ n dữ liệu. Khi dùng kĩ thuật này thì tập n dữ liệu mới của mình có thể có những dữ liệu bị trùng nhau.
            </div>
            """, unsafe_allow_html=True)
            
            st.image("imageB1/sampling.png", use_container_width=True)
            st.markdown("""
            <div style="margin-left: 40px; line-height: 1.5;">
                1. Sau khi sample được n dữ liệu từ bước 1 thì mình chọn ngẫu nhiên ở k thuộc tính (k < n). Giờ mình được bộ dữ liệu mới gồm n dữ liệu và mỗi dữ liệu có k thuộc tính.  
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin-left: 40px; line-height: 1.5;">    
                2. Dùng thuật toán Decision Tree để xây dựng cây quyết định với bộ dữ liệu ở bước 2.

            </div>
            """, unsafe_allow_html=True)

            st.write("""   
            Do quá trính xây dựng mỗi cây quyết định đều có yếu tố ngẫu nhiên (random) nên kết quả là các cây quyết định trong thuật toán Random Forest có thể khác nhau.

            Thuật toán Random Forest sẽ bao gồm nhiều cây quyết định, mỗi cây được xây dựng dùng thuật toán Decision Tree trên tập dữ liệu khác nhau và dùng tập thuộc tính khác nhau. Sau đó kết quả dự đoán của thuật toán Random Forest sẽ được tổng hợp từ các cây quyết định.

            Khi dùng thuật toán Random Forest, mình hay để ý các thuộc tính như: số lượng cây quyết định sẽ xây dựng, số lượng thuộc tính dùng để xây dựng cây. Ngoài ra, vẫn có các thuộc tính của thuật toán Decision Tree để xây dựng cây như độ sâu tối đa, số phần tử tối thiểu trong 1 node để có thể tách.

            ### 3. Công thức toán học
            Với bài toán phân loại, kết quả dự đoán cuối cùng được tính như sau:
            """)

            st.latex(r'''
            \hat{y} = \text{mode} \{ h_1(x), h_2(x), \dots, h_T(x) \}
            ''')
            st.write("Trong đó:")
            st.markdown("""
            - $\\hat{y}$: Lớp dự đoán cuối cùng.
            - $h_t(x)$: Dự đoán của cây thứ $t$ cho đầu vào $x$.
            - $T$: Số lượng cây trong rừng.
            - $\\text{mode}$: Hàm lấy giá trị xuất hiện nhiều nhất (phiếu bầu đa số).
            """)
            st.write("""
            Với bài toán hồi quy:
            """)

            st.latex(r'''
            \hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(x)
            ''')

            st.write("Trong đó:")
            st.markdown("""
            - $\\hat{y}$: Giá trị dự đoán trung bình.
            - $h_t(x)$: Dự đoán của cây thứ t cho đầu vào x.
            - $T$: Số lượng cây trong rừng.
            """)
            st.write("""
            ### 4. Ưu điểm và nhược điểm
            #### Ưu điểm:
            
            + **Khả năng tổng quát hóa tốt:** Nhờ yếu tố ngẫu nhiên, Random Forest giảm thiểu hiện tượng overfitting so với một cây quyết định đơn lẻ.
            + **Khả năng xử lý dữ liệu lớn:** Random Forest hoạt động hiệu quả trên dữ liệu có số lượng lớn mẫu và đặc trưng.
            + **Đơn giản và linh hoạt:** Không cần nhiều siêu tham số tinh chỉnh như một số mô hình khác.

            #### Nhược điểm:
            - **Tốn tài nguyên:** Do cần huấn luyện nhiều cây quyết định, Random Forest có thể yêu cầu nhiều tài nguyên tính toán và bộ nhớ hơn so với một cây quyết định đơn.
            - **Khó diễn giải:** Kết quả của Random Forest khó giải thích hơn so với một cây quyết định đơn vì nó là tập hợp nhiều cây.
            """)

        def data():
            st.write("Chọn tệp dữ liệu và số dòng cần hiển thị:")
            
            if datasets:
                selected_file = st.selectbox("Chọn tệp dữ liệu", list(datasets.keys()))
                num_rows = st.slider("Số dòng cần hiển thị", min_value=1, max_value=len(datasets[selected_file]), value=5)
                st.write(datasets[selected_file].head(num_rows))
            else:
                st.write("Chưa có tệp dữ liệu nào được tải lên.")

        def xu_ly_du_lieu():
            if "processed_datasets" not in st.session_state:
                st.session_state.processed_datasets = {}

            if not datasets:
                st.write("🚫 Chưa có tệp dữ liệu nào được tải lên.")
                return

            selected_file = st.selectbox("📂 Chọn tệp dữ liệu để xử lý", list(datasets.keys()))
            df = datasets[selected_file].copy()
            missing_data = df.isnull().sum()

            # Lọc ra các cột có dữ liệu thiếu
            missing_cols = missing_data[missing_data > 0].index.tolist()

            st.write("📊 Tổng số giá trị thiếu trong từng cột:")
            st.write(missing_data)

            if missing_cols:
                st.info(f"🔍 Các cột có dữ liệu thiếu: {', '.join(missing_cols)}")
            else:
                st.success("✅ Không có cột nào bị thiếu dữ liệu.")

            columns_to_process = st.multiselect("🔍 Chọn các cột để xử lý", df.columns)
            
            needs_update = False
            cols_to_drop = []

            for col in columns_to_process:
                if col in missing_cols:
                    st.warning(f"⚠️ Cột '{col}' có {missing_data[col]} giá trị thiếu. Chọn phương pháp xử lý:")
                    
                    # Gợi ý về cách xử lý
                    st.write("- **Xóa cột**: Khi dữ liệu thiếu quá nhiều hoặc cột không quan trọng.")
                    st.write("- **Thay thế bằng giá trị trung bình**: Khi cột là số và dữ liệu có phân phối chuẩn.")
                    st.write("- **Thay thế bằng giá trị trung vị**: Khi có ngoại lệ hoặc dữ liệu bị lệch.")
                    st.write("- **Thay thế bằng giá trị phổ biến nhất**: Khi dữ liệu là danh mục.")

                    method = st.selectbox(f"🔧 Xử lý '{col}'", [
                        "Không thay đổi", "Xóa cột",
                        "Thay thế bằng giá trị trung bình", 
                        "Thay thế bằng giá trị trung vị", "Thay thế bằng giá trị phổ biến nhất"
                    ], key=f"method_{col}")
                    
                    if method == "Xóa cột":
                        df.drop(columns=[col], inplace=True)  # Xóa toàn bộ cột
                        needs_update = True
                    elif method == "Thay thế bằng giá trị trung bình":
                        df[col].fillna(round(df[col].mean(), 0), inplace=True)
                        needs_update = True
                    elif method == "Thay thế bằng giá trị trung vị":
                        df[col].fillna(df[col].median(), inplace=True)
                        needs_update = True
                    elif method == "Thay thế bằng giá trị phổ biến nhất":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        needs_update = True
                else:
                    # Cảnh báo nếu chọn cột không thiếu dữ liệu
                    st.warning(f"⚠️ Cột '{col}' không có dữ liệu thiếu. Chỉ có thể xóa cột.")
                    confirm_delete = st.checkbox(f"❌ Bạn có chắc muốn xóa cột '{col}'?", key=f"confirm_{col}")

                    if confirm_delete:
                        cols_to_drop.append(col)

            if st.button("✅ Xử lý"):
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
                    st.success(f"✅ Đã xóa các cột: {', '.join(cols_to_drop)}")

                if needs_update or cols_to_drop:
                    st.session_state.processed_datasets[selected_file] = df.copy()
                
                st.write("📌 Dữ liệu sau khi xử lý:")
                st.write(df)

        def process_and_split_data():
            if "normalized_datasets" not in st.session_state:
                st.session_state.normalized_datasets = {}

            if "processed_datasets" not in st.session_state or not st.session_state.processed_datasets:
                st.write("🚫 Chưa có dữ liệu đã xử lý.")
                return

            selected_file = st.selectbox("📂 Chọn dữ liệu để chuẩn hóa và chia", list(st.session_state.processed_datasets.keys()))
            df = st.session_state.processed_datasets[selected_file].copy()


  

            st.write("📊 **Dữ liệu trước khi chuẩn hóa:**")
            st.dataframe(df)
            st.write("""
            - Hệ thống sẽ tự động ánh xạ các cột có  <10 giá trị khác nhau trong một cột không thuộc kiển số
            - Hệ thống sẽ tự động xóa các cột >= 10 giá trị khác nhau trong cùng 1 cột không thuộc kiểu số
            - Sau đó chuẩn hóa tất cả về đoạn [0;1]
            """)
            if df.isnull().sum().sum() > 0:
                st.warning("⚠️ Dữ liệu vẫn còn giá trị thiếu. Vui lòng xử lý trước khi tiếp tục.")
                return

            # Xử lý các cột không phải số
            cols_to_drop = []
            for col in df.select_dtypes(exclude=['number']).columns:
                unique_values = df[col].nunique()
                if unique_values <= 10:
                    mapping = {val: idx+1 for idx, val in enumerate(df[col].unique())}
                    df[col] = df[col].map(mapping)
                else:
                    cols_to_drop.append(col)

            # Xóa các cột có quá nhiều giá trị khác nhau
            df.drop(columns=cols_to_drop, inplace=True)

            # Chuẩn hóa dữ liệu số
            numeric_df = df.select_dtypes(include=['number']).copy()
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

            # Giữ nguyên số lượng hàng
            df_scaled.index = df.index  
            st.session_state.normalized_datasets[selected_file] = df_scaled
            st.session_state.scaler = scaler  # Lưu bộ scaler để dùng lại trong demo()

            st.write("✅ **Dữ liệu sau khi chuẩn hóa:**")
            st.dataframe(df_scaled.head())

            # Điều chỉnh % train, test từ 0-100%
            train_ratio = st.slider(' %Train', 0, 100, 70) / 100
            test_ratio = st.slider(' %Test', 0, 100, 15) / 100

            # Đảm bảo tổng train + test không vượt 100%
            if train_ratio + test_ratio > 1.0:
                test_ratio = 1.0 - train_ratio

            val_ratio = test_ratio  # Validation = phần còn lại của test

            # Chia dữ liệu
            train_df, temp_df = train_test_split(df_scaled, train_size=train_ratio, random_state=42)
            if test_ratio > 0 and val_ratio > 0:
                test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            elif test_ratio > 0:
                test_df, val_df = temp_df, None
            else:
                test_df, val_df = None, None

            st.session_state.train_data = train_df
            st.session_state.test_data = test_df
            st.session_state.val_data = val_df

            # Hiển thị số lượng mẫu sau khi chia
            st.write(f"📊 Số mẫu Train: {len(train_df)}")
            st.write(f"📊 Số mẫu Test: {len(test_df) if test_df is not None else 0}")
            st.write(f"📊 Số mẫu Validation: {len(val_df) if val_df is not None else 0}")
            st.success("✅ Dữ liệu đã được chuẩn hóa và chia thành công!")

        def train():
            if "train_data" not in st.session_state or st.session_state.train_data is None:
                st.write("🚫 Chưa có dữ liệu train. Vui lòng xử lý dữ liệu trước.")
                return
            
            # Lấy dữ liệu train và test từ session_state
            train_df = st.session_state.train_data
            test_df = st.session_state.test_data

            st.write("📌 Dữ liệu Train (5 dòng đầu):", train_df.head())

            # Chọn cột đầu ra (biến mục tiêu)
            target_col = st.selectbox("🎯 Chọn cột mục tiêu", train_df.columns)

            # Tách đầu vào (X) và đầu ra (y)
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col]) if test_df is not None else None
            y_test = test_df[target_col] if test_df is not None else None

            # Lựa chọn mô hình hồi quy
            regression_type = st.radio("📌 Chọn thuật toán hồi quy", ["Multiple Regression", "Polynomial Regression"])

            if regression_type == "Multiple Regression":
                model = LinearRegression()
            else:
                poly_degree = st.slider("🔢 Chọn bậc của đa thức", 2, 5, 2)  # Chọn bậc của đa thức (mặc định là 2)
                model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())

            # Thực hiện Cross Validation với 5-Fold
            st.write("🔄 Đang thực hiện Cross Validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')  # Hệ số R^2

            # Làm tròn giá trị Cross Validation
            cv_scores_rounded = [round(score, 2) for score in cv_scores]

            # Hiển thị kết quả
            st.write("📊 Kết quả Cross Validation (R^2 Score):", cv_scores_rounded)
            st.write("📈 Giá trị trung bình R^2:", round(cv_scores.mean(), 2))

            # Huấn luyện mô hình trên toàn bộ tập train
            model.fit(X_train, y_train)

            # Lưu mô hình vào session_state
            st.session_state.trained_model = model
            st.success(f"✅ Mô hình {regression_type} đã được huấn luyện xong!")
            st.session_state.feature_columns = X_train.columns.tolist()
            st.write("📌 Các cột đầu vào đã lưu:", st.session_state.feature_columns)    
        def demo():
            if "trained_model" not in st.session_state or st.session_state.trained_model is None:
                st.write("🚫 Chưa có mô hình được huấn luyện. Vui lòng huấn luyện mô hình trước.")
                return

            model = st.session_state.trained_model  # Lấy mô hình đã lưu

            if "feature_columns" not in st.session_state or st.session_state.feature_columns is None:
                st.write("⚠️ Không tìm thấy thông tin về các cột đầu vào. Vui lòng huấn luyện lại mô hình.")
                return

            feature_columns = st.session_state.feature_columns  # Lấy danh sách cột đầu vào

            if "test_data" in st.session_state and st.session_state.test_data is not None:
                test_df = st.session_state.test_data
            else:
                st.write("⚠️ Không có dữ liệu test. Vui lòng cung cấp dữ liệu đầu vào để dự đoán.")
                return

            # Kiểm tra target_column
            if "target_column" not in st.session_state:
                st.session_state.target_column = None

            # Lọc các cột đầu vào đúng theo feature_columns đã lưu
            input_columns = [col for col in feature_columns if col in test_df.columns]
            st.write("📌 Các cột đầu vào:", input_columns)

            # Tạo form nhập liệu
            user_input = {}
            for col in input_columns:
                default_value = test_df[col].mean()
                user_input[col] = st.number_input(f"🔢 Nhập giá trị cho '{col}'", value=default_value if pd.notna(default_value) else 0.0)

            # Nút bấm "Dự đoán"
            if st.button("🚀 Dự đoán"):
                # Chuyển dữ liệu đầu vào thành DataFrame
                input_df = pd.DataFrame([user_input])

                # Đảm bảo dữ liệu đầu vào có đúng cột như khi train
                input_df = input_df.reindex(columns=feature_columns, fill_value=0)

                st.write("📊 Dữ liệu đầu vào sau khi chuẩn hóa:")
                st.write(input_df.round(2))

                # Thực hiện dự đoán
                prediction = model.predict(input_df)[0]

                # Hiển thị kết quả dự đoán
                st.success(f"🎯 Dự đoán kết quả: {prediction:.2f}")

                # So sánh với giá trị thực tế trong tập test (nếu có target_column hợp lệ)
                if st.session_state.target_column and st.session_state.target_column in test_df.columns:
                    actual_value = test_df.iloc[0][st.session_state.target_column]
                    st.write(f"✅ Giá trị thực tế: {actual_value:.2f}")
                    error = abs(prediction - actual_value)
                    st.write(f"📉 Sai số: {error:.2f}")

                    if error < 0.1 * abs(actual_value):  # Nếu sai số nhỏ hơn 10% giá trị thực tế
                        st.success("✅ Dự đoán khá chính xác!")
                    else:
                        st.warning("⚠️ Dự đoán có sai số lớn, có thể cần cải thiện mô hình.")
                else:
                    st.write("⚠️ Không tìm thấy giá trị thực tế để kiểm tra độ chính xác.")

            if st.button("🚀 Dự đoán với MLflow"):
                if 'input_df' in locals() and input_df is not None:
                    prediction = mlflow_section(input_df)
                    st.success(f"🎯 Dự đoán từ MLflow: {prediction[0]:.2f}")  # Giả sử kết quả là mảng
                else:
                    st.error("❌ Lỗi: Dữ liệu đầu vào chưa được tạo.")                

       
        def mlflow_section(input_df):  
            st.error("❌ Lỗi: Dữ liệu đầu vào chưa được tạo.")    

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
            ly_thuyet_Random_Forest()

        with tab2:
            data()
            
        with tab3:
            xu_ly_du_lieu()

        with tab4:
            process_and_split_data()
            train()

        with tab5:
            demo()
        
        with tab6:
            # Nút bấm "Dự đoán với MLflow"
            mlflow_section()
    else:
        st.write("Vui lòng tải lên ít nhất một file dữ liệu để bắt đầu.")

if __name__ == "__main__":
    Classification()
