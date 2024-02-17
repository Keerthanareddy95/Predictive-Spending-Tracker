import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main


def main():
    st.title("End to End Predctive Spending Tracker with ZenML")
    st.markdown(
       """ 
    #### Description of Features 
    This app is designed to predict the customer expenses. You can input the features  listed below and get the spending prediction. 
    | Models        | Description   | 
    | ------------- | -     | 
    |Description | The description of the purchase made by the Customer.| 
    | Amount   | Amount paid by the customer in USD |  
    | Transaction Type | Wheter the amount is paid in Debit ot Credit mode | 
    | Category | The category of the purchase |
    | Month | The month in which the purchase is made.  | 
    | Year | The year in which the purchase is made. |
    """
    )
    Description = st.number_input("Description")
    Transaction_Type = st.number_input("Transaction_Type")
    Category = st.number_input("Category")
    Month = st.number_input("Month")
    Year = st.number_input("Year")
    
    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            main()

        df = pd.DataFrame(
            {
                "Description": [Description],
                "Transaction_Type": [Transaction_Type],
                "Category": [ Category],
                "Month": [Month],
                "Year": [Year],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your expenses prediction with given product details is :-{}".format(
                pred
            )
        )
    st.write("Your Expense prediction for the given feature is 44.68")



if __name__ == "__main__":
    main()