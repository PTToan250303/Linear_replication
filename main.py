# main.py


import streamlit as st
from streamlit_option_menu import option_menu
import buoi1.Linear_Regression as Linear_Regression
import buoi2.app as b2
import buoi3.app2 as b3
import buoi4.PCA_t_SNE as b4
def main():
    with st.sidebar:
        selected = option_menu(
            "Menu",
            ["Trang chủ", "Linear Regression","Assignment - Classification","Clustering Algorithms","PCA & t-SNE"],
            icons=["house"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Trang chủ":
        st.title("Chào mừng đến với ứng dụng của bạn!")
        st.write("Chọn một tùy chọn từ menu để tiếp tục.")

    elif selected == "Linear Regression":
        Linear_Regression.Classification()  # Gọi hàm từ buoi1.py
    elif selected == "Assignment - Classification":
        b2.Classification()  # Gọi hàm từ buoi2.py
    elif selected == "Clustering Algorithms":
        b3.main()  # Gọi hàm từ buoi3.py
    elif selected == "PCA & t-SNE":
        b4.pca_tsne() # Gọi hàm từ buoi4.py
if __name__ == "__main__":
    main()
