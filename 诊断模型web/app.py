import streamlit as st
import numpy as np
import joblib
import os

# 获取当前文件的路径
current_dir = os.path.dirname(__file__)

# 模型和标准化器的文件路径
model_path = os.path.join(current_dir, 'gbc_clf_model.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

# 加载模型和新的标准化器
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 所有特征名称
all_features = ['Height', 'Weight', 'IFNα', 'IFNβ', 'IL-10', 'IL-12p70', 'IL-17', 'Il-18', 'IL-1β', 'IL-2',
                'IL-2R', 'IL-4', 'IL-5', 'IL-6', 'IL-8', 'TNFα', 'TNFβ', 'SAA', 'PCT', 'ESR', 'SF',
                'LDH', 'CRP', 'Fever', 'Hotpeak', 'Cough', 'Gasp', 'Spo2', 'Age', 'Sex_Female', 'Sex_Male']

# 最终模型训练使用的特征名称
selected_features = ['TNFβ', 'Height', 'Age', 'IL-2', 'ESR', 'Fever', 'PCT', 'Cough', 'IFNα', 'IL-1β', 'IL-6', 'LDH']

# 初始化页面
st.title('The diagnostic model for Mycoplasma pneumoniae infection')

# 创建用户输入控件
TNF_beta = st.number_input('TNFβ(pg/ml)', value=0.0)
Height = st.number_input('Height(cm)', value=0.0)
Age = st.number_input('Age(Months)', value=0.0)
IL_2_value = st.number_input('IL-2(U/ml)', value=0.0)
ESR = st.number_input('ESR(mm/h)', value=0.0)
Fever = st.number_input('Fever(days)', value=0.0)
PCT = st.number_input('PCT(ng/ml)', value=0.0)
Cough = st.number_input('Cough(days)', value=0.0)
IFN_alpha = st.number_input('IFNα(pg/ml)', value=0.0)
IL_1_beta = st.number_input('IL-1β(pg/ml)', value=0.0)
IL_6 = st.number_input('IL-6(pg/ml)', value=0.0)
LDH = st.number_input('LDH(U/ml)', value=0.0)

# 将输入特征存储在字典中，并将未使用的特征填充为0
input_features = {
    'TNFβ': TNF_beta,
    'Height': Height,
    'Age': Age,
    'IL-2': IL_2_value,
    'ESR': ESR,
    'Fever': Fever,
    'PCT': PCT,
    'Cough': Cough,
    'IFNα': IFN_alpha,
    'IL-1β': IL_1_beta,
    'IL-6': IL_6,
    'LDH': LDH
}

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
    st.markdown("The reference cutoff value is 56.7%. At this cutoff value, the sensitivity is 82.2% and the specificity is 82.5%.", unsafe_allow_html=True)



