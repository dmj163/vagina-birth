import streamlit as st
import pandas as pd
import joblib


# 页面内容设置
# 页面名称
st.set_page_config(page_title="vaginal birth", layout="wide")
# 标题
st.title('An  explainable machining learning model in predicting vaginal birth after cesarean section')

st.markdown('This is an online tool to predict the success of vaginal birth after cesarean section by using machining learning model.\
         Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction.')

st.markdown('## Input Data:')
# 隐藏底部水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)


@st.cache
def predict_quality(model, df):
    y_pred = model.predict_proba(df)
    return y_pred[:, 1]


def option_name1(x):
    if x == 1:
        return "illiteracy"
    if x == 2:
        return 'primary school'
    if x == 3:
        return 'middle school'
    if x == 4:
        return 'high school/polytechnic school'
    if x == 5:
        return 'college'
    if x == 6:
        return 'postgraduate'

# 导入模型
model = joblib.load('cbEFW.pkl')

st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
EDUC = st.sidebar.selectbox(label='Education level', options=[1, 2, 3, 4, 5, 6],
                                     format_func=lambda x: option_name1(x), index=0)

PREBMI = st.sidebar.slider(label='BMI', min_value=1.00,
                        max_value=100.00,
                        value=24.00,
                        step=0.01)

Interval_of_pregnancy = st.sidebar.slider(label='Interval of pregnancy', min_value=1,
                        max_value=200,
                        value=80,
                        step=1)

BISHOP = st.sidebar.number_input(label='BISHOP', min_value=1.0,
                              max_value=30.0,
                              value=3.0,
                              step=1.0)

EFW = st.sidebar.number_input(label='EFW', min_value=0,
                              max_value=5000,
                              value=100,
                              step=10)


features = {'PREBMI': PREBMI,
            'Interval of pregnancy': Interval_of_pregnancy,
            'BISHOP': BISHOP,
            'EDUC': EDUC,
            'EFW': EFW,
            }

features_df = pd.DataFrame([features])
# 显示输入的特征
st.table(features_df)

# 显示预测结果与shap解释图
if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write("the probability of vaginal birth after cearean section:")
    st.success(round(prediction[0], 3))
