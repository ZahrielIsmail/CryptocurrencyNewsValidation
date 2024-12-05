import streamlit as st

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction Link (Ethereum)","Prediction Link (Cardano)", "Prediction Link (Shib)", "Information"])

# Load the selected page
if page == "Home":
    st.write("# Welcome to the Crypto Prediction App")
    st.write("Use the sidebar to navigate to different sections.")
    st.write("""
    This application utilizes advanced machine learning models to predict the performance of three different cryptocurrencies: Ethereum, Cardano, and Shiba Inu. 
    Each token is analyzed using three distinct models, providing a comprehensive forecast based on various metrics.

    **Navigate to the Information page** to learn more about the models used and their performance on each cryptocurrency. Here, you'll find detailed descriptions and insights that highlight the strengths and unique aspects of each model.
    """)

elif page == "Prediction Link (Ethereum)":
    import prediction_link_ethereum
    prediction_link_ethereum.run()
elif page == "Prediction Link (Cardano)":
    import prediction_link_cardano
    prediction_link_cardano.run()
elif page == "Prediction Link (Shib)":
    import prediction_link_shib
    prediction_link_shib.run()
elif page == "Information":
    st.write("# Model Information")
    st.write("This section provides information about the models used in this app.")

    # Ethereum Information
    st.image("Ethereum_Performance.png", caption="Ethereum Model Performance", use_container_width=True)
    st.write("### Ethereum:")
    st.write("- **Best Model:** DT + GB")
    st.write("- **Performance:**")
    st.write("  - MSE: 0.494")
    st.write("  - RMSE: 0.703")
    st.write("  - MAE: 0.372")
    st.write("  - R-Squared: 0.618")
    st.write("- **Insight:** The inclusion of GB in the best models suggests that Ethereum data contains complex, non-linear relationships, poorly captured by simpler models like LR.")

    # Cardano Information
    st.image("Cardano_Performance.png", caption="Cardano Model Performance", use_container_width=True)
    st.write("### Cardano:")
    st.write("- **Best Model:** DT + GB")
    st.write("- **Performance:**")
    st.write("  - MSE: 1.781")
    st.write("  - RMSE: 1.334")
    st.write("  - MAE: 0.946")
    st.write("  - R-Squared: -0.073")
    st.write("- **Insight:** High errors across models and the negative R-squared value indicate significant modeling challenges, such as high complexity and noise in Cardano data.")

    # Shiba Inu Information
    st.image("Shiba_Inu_Performance.png", caption="Shiba Inu Model Performance", use_container_width=True)
    st.write("### Shiba Inu:")
    st.write("- **Best Model:** LR + DT")
    st.write("- **Performance:**")
    st.write("  - MSE: 0.599")
    st.write("  - RMSE: 0.774")
    st.write("  - MAE: 0.540")
    st.write("  - R-Squared: 0.669")
    st.write("- **Insight:** The best model indicates that Shiba Inu data benefits from models capturing both linear and non-linear relationships, highlighting its complex patterns.")



# Save your images in the same directory or provide the correct path to them

# If you have images, place them in the same directory as your app script or provide the full path to the images
