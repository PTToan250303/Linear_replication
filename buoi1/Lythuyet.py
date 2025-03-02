import streamlit as st 
def ly_thuyet_Random_Forest():
            st.title("Random Forest")

            st.write("""
            Random forest là thuật toán supervised learning, có thể giải quyết cả bài toán regression và classification.
                     
            ### 1. Giới thiệu
            Random là ngẫu nhiên, Forest là rừng, nên ở thuật toán Random Forest mình sẽ xây dựng nhiều cây quyết định bằng thuật toán Decision Tree, tuy nhiên mỗi cây quyết định sẽ khác nhau (có yếu tố random). Sau đó kết quả dự đoán được tổng hợp từ các cây quyết định.
            
            Ở bước huấn luyện thì mình sẽ xây dựng nhiều cây quyết định, các cây quyết định có thể khác nhau (phần sau mình sẽ nói mỗi cây được xây dựng như thế nào).
            """)
            st.image("random_forest.png", use_container_width=True)
            st.write("""
            Sau đó ở bước dự đoán, với một dữ liệu mới, thì ở mỗi cây quyết định mình sẽ đi từ trên xuống theo các node điều kiện để được các dự đoán, sau đó kết quả cuối cùng được tổng hợp từ kết quả của các cây quyết định.            
            """)

            st.image("random_forest_predict.png", use_container_width=True)
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
            
            st.image("sampling.png", use_container_width=True)
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

if __name__ == "__main__":
    ly_thuyet_Random_Forest()