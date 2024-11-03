import streamlit as st
from pages import home, dashboard, batch


def main():
    st.set_page_config(
        page_title="Medicare Fraud Detection",
        page_icon="🏥",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Batch Predictions"])

    if page == "Home":
        home.show()
    elif page == "Dashboard":
        dashboard.show()
    else:
        batch.show()


if __name__ == "__main__":
    main()
