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

def run_logistic_regression_analysis(df):
    st.title('Logistic 回歸分析工具 (單變量 & 多變量)')

    st.write("數據預覽：")
    st.write(df.head())

    target_column = st.selectbox("選擇目標變量", df.columns)
    feature_columns = st.multiselect("選擇自變量", [col for col in df.columns if col != target_column])

    if feature_columns:
        st.write("選定變量的描述性統計：")
        st.write(df[feature_columns + [target_column]].describe())

        st.write("相關性矩陣：")
        corr_matrix = df[feature_columns + [target_column]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        X = df[feature_columns]
        y = df[target_column]

        # 檢查目標變量是否為二元變量
        unique_values = y.unique()
        if len(unique_values) != 2:
            st.error(f"目標變量 '{target_column}' 不是二元變量。請選擇一個只有兩個唯一值的變量，或者設置一個閾值來將其轉換為二元變量。")
            threshold = st.number_input(f"請輸入 {target_column} 的閾值來創建二元變量", value=y.mean())
            y = (y > threshold).astype(int)
            st.write(f"已將 {target_column} 轉換為二元變量。大於 {threshold} 的值設為 1，其他設為 0。")

        imputation_method = st.selectbox(
            "選擇缺失值處理方法",
            ["刪除缺失值", "均值填充", "中位數填充", "KNN填充", "多重插補", "缺失值指示器"]
        )

        if st.button("運行分析"):
            X_imputed = handle_missing_values(X, imputation_method)
            run_logistic_regression(X_imputed, y)

def handle_missing_values(X, method):
    if method == "刪除缺失值":
        return X.dropna()
    elif method == "均值填充":
        imputer = SimpleImputer(strategy='mean')
    elif method == "中位數填充":
        imputer = SimpleImputer(strategy='median')
    elif method == "KNN填充":
        imputer = KNNImputer(n_neighbors=2)
    elif method == "多重插補":
        imputer = IterativeImputer(random_state=0)
    elif method == "缺失值指示器":
        for col in X.columns:
            X[f'{col}_missing'] = X[col].isnull().astype(int)
        imputer = SimpleImputer(strategy='mean')
    
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imputed

def run_logistic_regression(X, y):
    st.write("## 單變量 Logistic 回歸分析")
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
    
    st.write("## 多變量 Logistic 回歸分析")
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
    st.write("變量統計信息：")
    st.write(odds_ratios_sorted)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=odds_ratios_sorted['p-value'], y=odds_ratios_sorted.index, ax=ax)
    plt.title("變量 p 值 (多變量分析)")
    plt.tight_layout()
    st.pyplot(fig)
    
    significant_predictors = odds_ratios_sorted[odds_ratios_sorted['p-value'] < 0.05]
    st.write("顯著的預測因子 (p < 0.05)：")
    st.write(significant_predictors)
    
    st.write("## 單變量 vs 多變量分析比較")
    comparison = pd.merge(univariate_results, odds_ratios_sorted, left_on='Variable', right_index=True, suffixes=('_univariate', '_multivariate'))
    st.write(comparison)

if __name__ == "__main__":
    run_logistic_regression_analysis(pd.DataFrame())
