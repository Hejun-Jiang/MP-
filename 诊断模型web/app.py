import streamlit as st
import numpy as np
import joblib

# 定义所有特征名称
all_features = ['Height', 'Weight', 'IFNα', 'IFNβ', 'IL-10', 'IL-12p70', 'IL-17', 'Il-18', 'IL-1β', 'IL-2',
                'IL-2R', 'IL-4', 'IL-5', 'IL-6', 'IL-8', 'TNFα', 'TNFβ', 'SAA', 'PCT', 'ESR', 'SF',
                'LDH', 'CRP', 'Fever', 'Hotpeak', 'Cough', 'Gasp', 'Spo2', 'Age', 'Sex_Female', 'Sex_Male']

# 初始化页面
st.title('The diagnostic model for Mycoplasma pneumoniae infection')

# 输入年龄
Age = st.number_input('Age (Months)', value=0.0)

# 定义特征和模型路径
if Age >= 60:
    model_path = 'et_model.pkl'
    scaler_path = 'scaler_over60.pkl'
    selected_features = ['TNFβ', 'IL-2', 'Fever', 'PCT', 'ESR', 'IL-2R', 'IFNα', 'Cough', 'IL-1β', 'Hotpeak', 'Il-18', 'Height', 'Weight', 'IL-10', 'IL-17', 'LDH']
    cutoff = 0.69
    sensitivity = 0.79
    specificity = 0.8
else:
    model_path = 'gbc_model.pkl'
    scaler_path = 'scaler_under60.pkl'
    selected_features = ['TNFβ', 'IL-2', 'Age', 'SF', 'PCT', 'ESR', 'Height', 'IFNβ', 'IL-5', 'IL-1β', 'Cough']
    cutoff = 0.34
    sensitivity = 0.855
    specificity = 0.833

if Age > 0:
    # 加载模型和标准化器
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 创建用户输入控件
    input_features = {}
    for feature in selected_features:
        input_features[feature] = st.number_input(f'{feature}', value=0.0)

    # 确保所有特征都存在，未使用的特征填充为0
    for feature in all_features:
        if feature not in input_features:
            input_features[feature] = 0.0

    # 构建包含所有特征的数组
    features = np.array([[input_features[feature] for feature in all_features]])

    # 标准化特征值
    scaled_features = scaler.transform(features)

    # 仅选择模型所需的特征进行预测
    scaled_features_selected = scaled_features[:, [all_features.index(feature) for feature in selected_features]]

    # 预测
    predicted_probs = model.predict_proba(scaled_features_selected)
    aki_probability = predicted_probs[0][1]

    if st.button('Diagnose'):
        st.markdown(f"<h3>Based on the feature values, the probability of diagnosing Mycoplasma pneumonia infection is <span style='color:red;'>{aki_probability * 100:.2f}%</span></h3>", unsafe_allow_html=True)
        st.markdown(f"The reference cutoff value is {cutoff * 100:.1f}%. At this cutoff value, the sensitivity is {sensitivity * 100:.1f}% and the specificity is {specificity * 100:.1f}%.", unsafe_allow_html=True)


