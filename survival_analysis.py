import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.impute import SimpleImputer

def run_survival_analysis(df):
    st.title('生存分析工具')

    st.write("數據預覽：")
    st.write(df.head())

    # 選擇時間、事件和協變量
    time_column = st.selectbox("選擇時間列", df.columns)
    event_column = st.selectbox("選擇事件列", df.columns)
    covariates = st.multiselect("選擇協變量", [col for col in df.columns if col not in [time_column, event_column]])

    if not covariates:
        st.warning("請至少選擇一個協變量進行分析。")
        return

    # 處理缺失值
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # 準備數據
    X = df_imputed[covariates]
    T = df_imputed[time_column]
    E = df_imputed[event_column]

    if st.button("運行分析"):
        # 單變量 Cox 回歸
        st.subheader("單變量 Cox 回歸分析")
        univariate_results = []
        for covariate in covariates:
            cph = CoxPHFitter()
            cph.fit(df_imputed[[time_column, event_column, covariate]], duration_col=time_column, event_col=event_column)
            summary = cph.summary
            univariate_results.append({
                'Variable': covariate,
                'HR': summary['exp(coef)'][0],
                'Lower 95%': summary['exp(coef) lower 95%'][0],
                'Upper 95%': summary['exp(coef) upper 95%'][0],
                'p': summary['p'][0]
            })
        univariate_df = pd.DataFrame(univariate_results)
        st.write(univariate_df)

        # 多變量 Cox 回歸
        st.subheader("多變量 Cox 回歸分析")
        cph = CoxPHFitter()
        cph.fit(df_imputed[[time_column, event_column] + covariates], duration_col=time_column, event_col=event_column)
        st.write(cph.summary)

        # Kaplan-Meier 曲線
        st.subheader("Kaplan-Meier 生存曲線")
        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots()
        kmf.fit(T, E, label="KM estimate")
        kmf.plot(ax=ax)
        plt.title("Kaplan-Meier Estimate")
        st.pyplot(fig)

        # Log-rank 檢驗
        st.subheader("Log-rank 檢驗")
        categorical_covariates = [col for col in covariates if df_imputed[col].nunique() <= 5]
        if categorical_covariates:
            selected_covariate = st.selectbox("選擇分組變量進行 Log-rank 檢驗", categorical_covariates)
            groups = df_imputed[selected_covariate].unique()
            if len(groups) == 2:
                group1 = df_imputed[df_imputed[selected_covariate] == groups[0]]
                group2 = df_imputed[df_imputed[selected_covariate] == groups[1]]
                results = logrank_test(group1[time_column], group2[time_column], 
                                       group1[event_column], group2[event_column])
                st.write(f"Log-rank 檢驗 p 值: {results.p_value:.4f}")

                # 繪製分組 KM 曲線
                fig, ax = plt.subplots()
                for group in groups:
                    mask = df_imputed[selected_covariate] == group
                    kmf = KaplanMeierFitter()
                    kmf.fit(df_imputed[mask][time_column], df_imputed[mask][event_column], label=f'{selected_covariate}={group}')
                    kmf.plot(ax=ax)
                plt.title(f"Kaplan-Meier Estimate by {selected_covariate}")
                st.pyplot(fig)
            else:
                st.warning("Log-rank 檢驗需要恰好兩個組別。")
        else:
            st.warning("沒有適合進行 Log-rank 檢驗的分類變量。")

# 如果直接運行此腳本，執行一個簡單的測試
if __name__ == "__main__":
    # 創建一個示例數據集
    np.random.seed(42)
    N = 1000
    df = pd.DataFrame({
        'time': np.random.exponential(50, size=N),
        'event': np.random.binomial(1, 0.7, size=N),
        'age': np.random.normal(65, 10, size=N),
        'sex': np.random.binomial(1, 0.5, size=N),
        'treatment': np.random.binomial(1, 0.5, size=N)
    })
    run_survival_analysis(df)
