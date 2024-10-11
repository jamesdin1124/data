import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def run_logistic_regression_analysis():
    st.title('Logistic 回归分析工具 (单变量 & 多变量)')

    uploaded_file = st.file_uploader("选择 CSV 文件", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("数据预览：")
        st.write(data.head())

        target_column = st.selectbox("选择目标变量", data.columns)
        feature_columns = st.multiselect("选择自变量", [col for col in data.columns if col != target_column])

        if feature_columns:
            st.write("选定变量的描述性统计：")
            st.write(data[feature_columns + [target_column]].describe())

            st.write("相关性矩阵：")
            corr_matrix = data[feature_columns + [target_column]].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            X = data[feature_columns]
            y = data[target_column]

            imputation_method = st.selectbox(
                "选择缺失值处理方法",
                ["删除缺失值", "均值填充", "中位数填充", "KNN填充", "多重插补", "缺失值指示器"]
            )

            if st.button("运行分析"):
                X_imputed = handle_missing_values(X, imputation_method)
                run_logistic_regression(X_imputed, y)

def handle_missing_values(X, method):
    if method == "删除缺失值":
        return X.dropna()
    elif method == "均值填充":
        imputer = SimpleImputer(strategy='mean')
    elif method == "中位数填充":
        imputer = SimpleImputer(strategy='median')
    elif method == "KNN填充":
        imputer = KNNImputer(n_neighbors=2)
    elif method == "多重插补":
        imputer = IterativeImputer(random_state=0)
    elif method == "缺失值指示器":
        for col in X.columns:
            X[f'{col}_missing'] = X[col].isnull().astype(int)
        imputer = SimpleImputer(strategy='mean')
    
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imputed

def run_logistic_regression(X, y):
    st.write("## 单变量 Logistic 回归分析")
    univariate_results = []
    
    for column in X.columns:
        X_univariate = sm.add_constant(X[[column]])
        model = sm.Logit(y, X_univariate)
        results = model.fit()
        
        conf_int = results.conf_int()
        odds_ratio = np.exp(results.params[column])
        p_value = results.pvalues[column]
        ci_lower = np.exp(conf_int.loc[column, 0])
        ci_upper = np.exp(conf_int.loc[column, 1])
        
        univariate_results.append({
            'Variable': column,
            'Odds Ratio': odds_ratio,
            'p-value': p_value,
            '2.5% CI': ci_lower,
            '97.5% CI': ci_upper
        })
    
    univariate_results = pd.DataFrame(univariate_results)
    univariate_results_sorted = univariate_results.sort_values('p-value')
    st.write(univariate_results_sorted)
    
    st.write("## 多变量 Logistic 回归分析")
    X_multivariate = sm.add_constant(X)
    model = sm.Logit(y, X_multivariate)
    results = model.fit()
    
    st.write("模型摘要：")
    st.text(results.summary())
    
    conf_int = results.conf_int()
    odds_ratios = pd.DataFrame({
        'Odds Ratio': np.exp(results.params),
        'p-value': results.pvalues,
        '2.5% CI': np.exp(conf_int[0]),
        '97.5% CI': np.exp(conf_int[1])
    })
    
    odds_ratios_sorted = odds_ratios.sort_values('p-value')
    st.write("变量统计信息：")
    st.write(odds_ratios_sorted)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=odds_ratios_sorted['p-value'], y=odds_ratios_sorted.index, ax=ax)
    plt.title("变量 p 值 (多变量分析)")
    plt.tight_layout()
    st.pyplot(fig)
    
    significant_predictors = odds_ratios_sorted[odds_ratios_sorted['p-value'] < 0.05]
    st.write("显著的预测因子 (p < 0.05)：")
    st.write(significant_predictors)
    
    st.write("## 单变量 vs 多变量分析比较")
    comparison = pd.merge(univariate_results, odds_ratios_sorted, left_on='Variable', right_index=True, suffixes=('_univariate', '_multivariate'))
    st.write(comparison)

if __name__ == "__main__":
    run_logistic_regression_analysis()