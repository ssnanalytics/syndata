import pandas as pd
import numpy as np
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import streamlit as st 
from os import getcwd
text_file=st.file_uploader("Upload the Data File")
st.write("-------------------------")

if text_file is not None:
    df=pd.read_csv(text_file)
    dd_list=df.columns
    cat_cols=st.multiselect("Select the Categorical Columns",dd_list)
    num_cols=st.multiselect("Select the Numerical Columns",dd_list)
    Output_file=st.text_input('Enter Output File Name')
    s=st.number_input('Enter the Sample Size',1000)
    OP=Output_file + '.csv'
    sub=st.button('Submit')
    if sub:
        batch_size = 50
        epochs = 3
        learning_rate = 2e-4
        beta_1 = 0.5
        beta_2 = 0.9

        ctgan_args = ModelParameters(batch_size=batch_size,
                                    lr=learning_rate,
                                    betas=(beta_1, beta_2))

        train_args = TrainParameters(epochs=epochs)
        synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
        synth.fit(data=df, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)
        df_syn = synth.sample(s)
        df_syn.to_csv(OP)
        c=getcwd()    
        c=c + '/' + OP
        with open(c,"rb") as file:
            st.download_button(label=':blue[Download]',data=file,file_name=OP,mime="image/png") 
        st.success("Thanks for using the app !!!")
# st.markdown("""
# <style>


# /* Hide Streamlit Branding */
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}
# </style>
# """, unsafe_allow_html=True)