# main.py
import streamlit as st
from streamlit_option_menu import option_menu
import buoi1.Linear_Regression as Linear_Regression

def main():
    with st.sidebar:
        selected = option_menu(
            "Menu",
            ["Trang chủ", "Khám phá dữ liệu"],
            icons=["house", "database"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Trang chủ":
        st.title("Chào mừng đến với ứng dụng của bạn!")
        st.write("Chọn một tùy chọn từ menu để tiếp tục.")

    elif selected == "Linear Regression":
        Linear_Regression.Classification()  # Gọi hàm từ buoi1.py

if __name__ == "__main__":
    main()
