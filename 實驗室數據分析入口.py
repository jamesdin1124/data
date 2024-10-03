import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import base64
import io
from PIL import Image
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 載入數據
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"無法讀取文件：{e}")
            return None
    return None

# 計算年齡
def calculate_age(birth_date, test_date):
    birth_date = pd.to_datetime(birth_date)
    test_date = pd.to_datetime(test_date)
    return (test_date - birth_date).days / 365.25

# 其他函數保持不變...
def is_numeric(series):
    return pd.to_numeric(series, errors='coerce').notnull().any()

def analyze_numeric_data(df, group_column, numeric_columns):
    summary_stats = {}
    for column in numeric_columns:
        if not is_numeric(df[column]):
            continue
        
        valid_data = df[[group_column, column]].dropna()
        if valid_data.empty or len(valid_data[group_column].unique()) != 2:
            continue
        
        unique_groups = valid_data[group_column].unique()
        group_1 = pd.to_numeric(valid_data[valid_data[group_column] == unique_groups[0]][column], errors='coerce')
        group_0 = pd.to_numeric(valid_data[valid_data[group_column] == unique_groups[1]][column], errors='coerce')
        
        group_1 = group_1.dropna()
        group_0 = group_0.dropna()
        
        if len(group_1) > 1 and len(group_0) > 1:
            try:
                t_stat, t_p = ttest_ind(group_1, group_0, equal_var=False)
                u_stat, u_p = mannwhitneyu(group_1, group_0)
            
                summary_stats[column] = {
                    f'Mean Group {unique_groups[0]}': group_1.mean(),
                    f'SD Group {unique_groups[0]}': group_1.std(),
                    f'Median Group {unique_groups[0]}': group_1.median(),
                    f'IQR Group {unique_groups[0]}': group_1.quantile(0.75) - group_1.quantile(0.25),
                    f'Mean Group {unique_groups[1]}': group_0.mean(),
                    f'SD Group {unique_groups[1]}': group_0.std(),
                    f'Median Group {unique_groups[1]}': group_0.median(),
                    f'IQR Group {unique_groups[1]}': group_0.quantile(0.75) - group_0.quantile(0.25),
                    'T-test P-value': t_p,
                    'Mann-Whitney P-value': u_p
                }
            except Exception:
                pass
    
    return pd.DataFrame.from_dict(summary_stats, orient='index')

def create_distribution_plot(df, group_column, selected_column, t_p, u_p):
    fig = go.Figure()
    
    for group in df[group_column].unique():
        group_data = df[df[group_column] == group][selected_column]
        fig.add_trace(go.Box(y=group_data, name=str(group)))
    
    fig.update_layout(
        title=f"{selected_column} 在 {group_column} 两组中的分布<br>T-test p-value: {t_p:.4f}, Mann-Whitney p-value: {u_p:.4f}",
        xaxis_title=group_column,
        yaxis_title=selected_column,
        showlegend=False
    )
    
    return fig

def create_boxplot_trend(df, group_column, selected_column, age_column, interval):
    df['age_interval'] = (df[age_column] // interval) * interval
    
    fig = go.Figure()
    
    colors = ['blue', 'red']
    
    for i, group in enumerate(df[group_column].unique()):
        group_data = df[df[group_column] == group]
        
        fig.add_trace(
            go.Box(x=group_data['age_interval'], y=group_data[selected_column], 
                   name=f'Group {group}', marker_color=colors[i])
        )
    
    age_intervals = sorted(df['age_interval'].unique())
    p_values = []
    
    for interval in age_intervals:
        group_0 = df[(df[group_column] == df[group_column].unique()[0]) & (df['age_interval'] == interval)][selected_column].dropna()
        group_1 = df[(df[group_column] == df[group_column].unique()[1]) & (df['age_interval'] == interval)][selected_column].dropna()
        
        if len(group_0) > 0 and len(group_1) > 0:
            try:
                _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
                p_values.append(p_value)
            except Exception as e:
                print(f"Error calculating p-value for interval {interval}: {str(e)}")
                p_values.append(None)
        else:
            p_values.append(None)
    
    max_y = df[selected_column].max()
    y_range = df[selected_column].max() - df[selected_column].min()
    
    for i, p_value in enumerate(p_values):
        if p_value is not None and not np.isnan(p_value):
            fig.add_annotation(
                x=age_intervals[i],
                y=max_y + y_range * 0.05,
                text=f"p={p_value:.4f}",
                showarrow=False,
                font=dict(size=10)
            )
        else:
            fig.add_annotation(
                x=age_intervals[i],
                y=max_y + y_range * 0.05,
                text="N/A",
                showarrow=False,
                font=dict(size=10)
            )
    
    fig.update_layout(
        title=f"{selected_column} 隨年齡變化的趨勢 (Mann-Whitney U 檢驗)",
        xaxis_title=f"年齡 (間隔: {interval} 年)",
        yaxis_title=selected_column,
        boxmode='group',
        legend_title=group_column,
        height=600,
        yaxis=dict(range=[df[selected_column].min() - y_range * 0.1, max_y + y_range * 0.2])
    )
    
    return fig

# 連續變量分析功能
def continuous_variable_analysis():
    st.header("比較組間連續變量統計差異")
    
    uploaded_file = st.file_uploader("請上傳您的CSV文件", type="csv")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.subheader("數據預覽")
            st.dataframe(df.head())

            group_column = st.selectbox("選擇用於分組的列", df.columns)
            continuous_vars = st.multiselect("選擇連續變量", df.columns)
            age_column = st.selectbox("選擇年齡列", df.columns)
            age_interval = st.number_input("選擇年齡間隔 (年)", min_value=1, value=5)

            if st.button("開始分析"):
                unique_groups = df[group_column].dropna().unique()
                if len(unique_groups) != 2:
                    st.error(f"分組列 '{group_column}' 應該包含兩個唯一值，但實際包含 {len(unique_groups)} 個值。請選擇另一個分組列。")
                else:
                    if continuous_vars:
                        numeric_summary_df = analyze_numeric_data(df, group_column, continuous_vars)
                        if not numeric_summary_df.empty:
                            st.subheader("數值變量比較表")
                            st.dataframe(numeric_summary_df.style.format({col: '{:.4f}' for col in numeric_summary_df.columns}))
                            
                            st.subheader("變量分佈和趨勢可視化")
                            valid_columns = numeric_summary_df.index.tolist()
                            if valid_columns:
                                for selected_column in valid_columns:
                                    t_p = numeric_summary_df.loc[selected_column, 'T-test P-value']
                                    u_p = numeric_summary_df.loc[selected_column, 'Mann-Whitney P-value']
                                    
                                    st.subheader(f"{selected_column} 總體分佈")
                                    fig = create_distribution_plot(df, group_column, selected_column, t_p, u_p)
                                    st.plotly_chart(fig)
                                    
                                    st.subheader(f"{selected_column} 隨年齡變化的趨勢")
                                    trend_fig = create_boxplot_trend(df, group_column, selected_column, age_column, age_interval)
                                    st.plotly_chart(trend_fig)

                            csv = numeric_summary_df.to_csv(index=True)
                            st.download_button(
                                label="下載數值變量分析結果 CSV",
                                data=csv,
                                file_name="numeric_summary_statistics.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info("選擇的連續變量中沒有可以分析的有效數據。")
                    else:
                        st.info("未選擇連續變量進行分析。")
    else:
        st.info("請上傳一個CSV文件來開始分析。")

# 實驗室檢查趨勢圖功能
def lab_test_trend():
    st.header("單一病歷號實驗室檢查項目趨勢圖")

    uploaded_file = st.file_uploader("選擇CSV文件", type="csv")

    df = load_data(uploaded_file)

    if df is not None and not df.empty:
        st.write("數據預覽：")
        st.write(df.head())

        patient_ids = st.multiselect('選擇病歷號（可多選）', df['Hx_number'].unique())

        if patient_ids:
            patient_data = df[df['Hx_number'].isin(patient_ids)]
            lab_tests = st.multiselect('選擇實驗室檢查項目', df.columns.drop(['Hx_number', 'Birthday', 'blood_sampling_time']))

            for test in lab_tests:
                fig = go.Figure()

                for patient_id in patient_ids:
                    patient_test_data = patient_data[patient_data['Hx_number'] == patient_id]
                    test_data = patient_test_data[['Hx_number', 'Birthday', 'blood_sampling_time', test]].copy()
                    test_data['檢查結果'] = pd.to_numeric(test_data[test], errors='coerce')
                    
                    test_data['年齡'] = test_data.apply(lambda row: calculate_age(row['Birthday'], row['blood_sampling_time']), axis=1)
                    test_data = test_data.sort_values('年齡')

                    fig.add_trace(go.Scatter(x=test_data['年齡'], y=test_data['檢查結果'], 
                                             mode='lines+markers', name=f'病歷號 {patient_id}',
                                             connectgaps=True))

                fig.update_layout(title=f'{test} 趨勢圖', xaxis_title='年齡', yaxis_title='檢查結果')
                st.plotly_chart(fig)
        else:
            st.warning('請選擇至少一個病歷號')
    else:
        st.warning('請上傳CSV文件以開始分析。')

# 主應用
def main():
    st.set_page_config(page_title="實驗室數據分析", layout="wide")
    st.title("實驗室數據分析應用")

    analysis_type = st.sidebar.radio("選擇分析類型", ["比較兩組連續變量統計差異", "單一病歷號實驗室檢查項目趨勢圖"])

    if analysis_type == "比較兩組連續變量統計差異":
        continuous_variable_analysis()
    elif analysis_type == "單一病歷號實驗室檢查項目趨勢圖":
        lab_test_trend()

if __name__ == '__main__':
    main()

#cd /Users/mbpr/.cursor-tutor/projects/資料處理/
#streamlit run 實驗室數據分析入口.py