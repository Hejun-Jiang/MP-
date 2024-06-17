import streamlit as st
import numpy as np
import joblib
import os

# 定义所有特征名称和单位
all_features = {
    'Height': 'Height (cm)',
    'Weight': 'Weight (kg)',
    'IFNα': 'IFNα (pg/ml)',
    'IFNβ': 'IFNβ (pg/ml)',
    'IL-10': 'IL-10 (pg/ml)',
    'IL-12p70': 'IL-12p70 (pg/ml)',
    'IL-17': 'IL-17 (pg/ml)',
    'Il-18': 'Il-18 (pg/ml)',
    'IL-1β': 'IL-1β (pg/ml)',
    'IL-2': 'IL-2 (U/ml)',
    'IL-2R': 'IL-2R (pg/ml)',
    'IL-4': 'IL-4 (pg/ml)',
    'IL-5': 'IL-5 (pg/ml)',
    'IL-6': 'IL-6 (pg/ml)',
    'IL-8': 'IL-8 (pg/ml)',
    'TNFα': 'TNFα (pg/ml)',
    'TNFβ': 'TNFβ (pg/ml)',
    'SAA': 'SAA (mg/L)',
    'PCT': 'PCT (ng/ml)',
    'ESR': 'ESR (mm/h)',
    'SF': 'SF (ng/ml)',
    'LDH': 'LDH (U/ml)',
    'CRP': 'CRP (mg/L)',
    'Fever': 'Fever (days)',
    'Hotpeak': 'Peak temperature (℃)',
    'Cough': 'Cough (days)',
    'Gasp': 'Gasp (days)',
    'Spo2': 'Spo2 (%)',
    'Age': 'Age (Months)',
    'Sex_Female': 'Sex_Female',
    'Sex_Male': 'Sex_Male'
}

# 初始化页面
st.title('The diagnostic model for Mycoplasma pneumoniae infection')
st.markdown("<p><strong>Please note, all information entered must be collected on the <span style='color:red; font-size:20px;'>the day of the patient's visit</span>.</strong></p>", unsafe_allow_html=True)
# 设定文件路径
base_path = '/mount/src/mp-/诊断模型web/'  # 确保此路径为你项目的根路径

# 输入年龄
if 'age' not in st.session_state:
    st.session_state.age = 0.0
Age = st.number_input('Age (Months)', value=st.session_state.age, key='age_input')

if st.button('Submit Age'):
    st.session_state.age_submitted = True
    st.session_state.age = Age

if 'age_submitted' in st.session_state:
    # 根据年龄选择特征和模型路径
    if Age >= 60:
        model_path = os.path.join(base_path, 'et_model.pkl')
        scaler_path = os.path.join(base_path, 'scaler_over60.pkl')
        selected_features = ['TNFβ', 'IL-2', 'Fever', 'PCT', 'ESR', 'IL-2R', 'IFNα', 'Cough', 'IL-1β', 'Hotpeak', 'Il-18', 'Height', 'Weight', 'IL-10', 'IL-17', 'LDH']
        cutoff = 0.69
        sensitivity = 0.79
        specificity = 0.8
    else:
        model_path = os.path.join(base_path, 'gbc_model.pkl')
        scaler_path = os.path.join(base_path, 'scaler_under60.pkl')
        selected_features = ['TNFβ', 'IL-2', 'Age', 'SF', 'PCT', 'ESR', 'Height', 'IFNβ', 'IL-5', 'IL-1β', 'Cough']
        cutoff = 0.34
        sensitivity = 0.855
        specificity = 0.833

    # 加载模型和标准化器
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 创建输入特征的控件
    input_features = {}
    for feature in selected_features:
        label = all_features[feature]
        input_features[feature] = st.number_input(f'{label}', value=0.0, key=f'{feature}_input')

    # 确保所有特征都存在，未使用的特征填充为0
    for feature in all_features.keys():
        if feature not in input_features:
            input_features[feature] = 0.0

    # 构建包含所有特征的数组
    features = np.array([[input_features[feature] for feature in all_features.keys()]])

    # 标准化特征值
    scaled_features = scaler.transform(features)

    # 仅选择模型所需的特征进行预测
    scaled_features_selected = scaled_features[:, [list(all_features.keys()).index(feature) for feature in selected_features]]

    if st.button('Diagnose'):
        # 预测
        predicted_probs = model.predict_proba(scaled_features_selected)
        aki_probability = predicted_probs[0][1]

        st.session_state.diagnosed = True
        st.session_state.aki_probability = aki_probability

    if 'diagnosed' in st.session_state:
        aki_probability = st.session_state.aki_probability
        st.markdown(f"<h3>Based on the feature values, the probability of diagnosing Mycoplasma pneumonia infection is <span style='color:red;'>{aki_probability * 100:.2f}%</span></h3>", unsafe_allow_html=True)
        st.markdown(f"The reference cutoff value is {cutoff * 100:.2f}%. At this cutoff value, the sensitivity is {sensitivity * 100:.1f}% and the specificity is {specificity * 100:.1f}%.", unsafe_allow_html=True)


