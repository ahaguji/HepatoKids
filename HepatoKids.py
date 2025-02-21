import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 加载模型和标准化器
gbc_model = joblib.load('HepatoKids.pkl')

feature_names = ["ALT","AST","GGT","HBsAg","HCT","PCT"]

# 创建Streamlit应用
st.title('慢乙肝儿童显著性肝脏炎症预测')
st.write('本应用可以帮助您预测慢乙肝儿童显著性肝脏炎症的概率')
# 创建用户输入控件
ALT = st.number_input('ALT(IU/L)', min_value=0.00, max_value=1000.00, value=0.00, step=0.01, key='ALT')
AST = st.number_input('AST (IU/L)', min_value=0.00, max_value=1000.00, value=0.00, step=0.01, key='AST')
GGT = st.number_input('GGT (IU/L)', min_value=0.00, max_value=200.00, value=0.00, step=0.01, key='GGT')
HBsAg = st.number_input('HBsAg (IU/mL)', min_value=0.000, max_value=60000.000, value=0.000, step=0.001, key='HBsAg')
HCT = st.number_input('HCT (%)', min_value=0.00, max_value=100.00, value=0.00, step=0.01, key='HCT')
PCT = st.number_input('PCT (%)', min_value=0.00, max_value=100.00, value=0.00, step=0.01, key='PCT')


# 计算log10(HBsAg)值
log10_HBsAg = np.log10(HBsAg) if HBsAg > 0 else 0

feature_values = [ALT, AST, GGT, log10_HBsAg, HCT, PCT]
features = np.array([feature_values])

# 创建一个按钮进行预测
if st.button('预测'):
    # 检查是否所有输入都已经提供
    if ALT == 0 or AST == 0 or HBsAg == 0 or HCT ==0 or PCT == 0 or GGT == 0:
        st.write("请填写所有字段")
    else:
    # 获取用户输入并创建数据框
     user_data = pd.DataFrame({
        'ALT': [ALT],
        'AST': [AST],
        'GGT': [GGT],
        'HBsAg': [HBsAg],
        'HCT': [HCT],
        'PCT': [PCT]
    })
    
    # 进行预测
    prediction_prob = gbc_model.predict_proba(user_data,)[0, 1]
    
    # 显示预测结果
    st.write(f'该患儿肝脏显著性炎症的发生概率是: {prediction_prob * 100:.2f}%')
    # Generate advice based on prediction results    
    if prediction_prob >=0.50:        
        advice = (            f'根据我们的模型，预测概率大于 50% 的 CHB 儿童发生肝脏显著性炎症的风险较高。'            
                              f'该患儿肝脏显著性炎症的发生概率是 {prediction_prob * 100:.2f}%. '           
                              '虽然这只是一个估计值，但它表明该患者可能面临较大发生肝脏显著性炎症的风险。 '           
                              '建议该患儿尽快接受肝活检以进行进一步评估，并确保准确诊断和必要的治疗。 ' )    
    else:        
        advice = (            f'根据我们的模型，预测概率大于 50% 的 CHB 儿童发生肝脏显著性炎症的风险较高。 '       
                              f'该患儿肝脏显著性炎症的发生概率是 {prediction_prob * 100:.2f}%. '           
                              '然而，保持健康的生活方式仍然非常重要。'            
                              '建议定期检查以监测您的肝脏健康状况，如果您出现任何症状，请及时寻求医疗建议。'        )
    st.write(advice)
    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(gbc_model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

