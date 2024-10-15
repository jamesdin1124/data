import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import base64
import io
from PIL import Image
import os
import matplotlib.pyplot as plt
from datetime import datetime
from logisticR import run_logistic_regression_analysis
from survival_analysis import run_survival_analysis

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

# 修改連續變量分析功能
def continuous_variable_analysis(df):
    st.header("比較組間連續變量統計差異")
    
    st.subheader("數據預覽")
    st.dataframe(df.head())

    group_column = st.selectbox("選擇用分組的列", df.columns)
    
    # 判斷分組列是否為數值型
    is_numeric_group = pd.api.types.is_numeric_dtype(df[group_column])
    
    if is_numeric_group:
        group_method = st.selectbox("選擇分組方法", ["二分法", "自定義閾值"])
        if group_method == "二分法":
            median_value = df[group_column].median()
            st.write(f"{group_column} 的中位��為: {median_value}")
            df['group'] = (df[group_column] > median_value).astype(int)
            st.write("已將數據分為兩組：0（小於等於中位數）和 1（大於中位數）")
        else:
            threshold = st.number_input(f"輸入 {group_column} 的閾值", value=float(df[group_column].mean()))
            df['group'] = (df[group_column] > threshold).astype(int)
            st.write(f"已將數據分為兩組：0（小於等於{threshold}）和 1（大於{threshold}）")
        group_column = 'group'
    
    continuous_vars = st.multiselect("選擇連續變量", df.columns)
    age_column = st.selectbox("選擇年齡列", df.columns)
    age_interval = st.number_input("選擇年齡間隔 (年)", min_value=1, value=5)

    if st.button("開始分析"):
        unique_groups = df[group_column].dropna().unique()
        if len(unique_groups) != 2:
            st.error(f"分組列 '{group_column}' 應該包含兩個唯一值，但實際包含 {len(unique_groups)} 個值。請選擇另一個分組列或調整分組方法。")
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
                            
                            st.subheader(f"{selected_column} 隨年齡化的趨勢")
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

# 實驗室檢查趨勢圖功能
def lab_test_trend(df):
    st.header("單一病歷號實驗室檢查項趨勢圖")

    st.subheader("數據預覽")
    st.write(df.head())

    patient_ids = st.multiselect('選擇病歷號（可多選）', df['Hx_number'].unique())

    if patient_ids:
        patient_data = df[df['Hx_number'].isin(patient_ids)]
        
        # 讓用戶選擇 X 軸
        x_axis_options = [col for col in df.columns if col not in ['Hx_number', 'Birthday']]
        x_axis = st.selectbox('選擇 X 軸', x_axis_options, index=x_axis_options.index('blood_sampling_time') if 'blood_sampling_time' in x_axis_options else 0)
        
        lab_tests = st.multiselect('選擇實驗室檢查項目', [col for col in df.columns if col not in ['Hx_number', 'Birthday', x_axis]])

        for test in lab_tests:
            fig = go.Figure()

            for patient_id in patient_ids:
                patient_test_data = patient_data[patient_data['Hx_number'] == patient_id]
                test_data = patient_test_data[['Hx_number', x_axis, test]].copy()
                test_data['檢查結果'] = pd.to_numeric(test_data[test], errors='coerce')
                
                # 如果 X 軸是日期類型,轉換為日期格式
                if pd.api.types.is_datetime64_any_dtype(test_data[x_axis]) or x_axis == 'blood_sampling_time':
                    test_data[x_axis] = pd.to_datetime(test_data[x_axis])
                elif x_axis == 'Birthday':
                    # 如果選擇 Birthday 作為 X 軸,計算年齡
                    test_data['年齡'] = test_data.apply(lambda row: calculate_age(row['Birthday'], row['blood_sampling_time']), axis=1)
                    x_axis = '年齡'
                
                test_data = test_data.sort_values(x_axis)

                fig.add_trace(go.Scatter(x=test_data[x_axis], y=test_data['檢查結果'], 
                                         mode='lines+markers', name=f'病歷號 {patient_id}',
                                         connectgaps=True))

            fig.update_layout(title=f'{test} 趨勢圖', xaxis_title=x_axis, yaxis_title='檢查結果')
            st.plotly_chart(fig)
    else:
        st.warning('選擇至少一個病歷號')

def calculate_age(birth_date, test_date):
    birth_date = pd.to_datetime(birth_date)
    test_date = pd.to_datetime(test_date)
    return (test_date - birth_date).days / 365.25

# 添加新的卡方分析函數
def add_totals_and_percentages(contingency_table):
    contingency_table['總計'] = contingency_table.sum(axis=1)
    total_row = contingency_table.sum().rename('總計')
    contingency_table = pd.concat([contingency_table, total_row.to_frame().T])
    
    # 計算每個目標列組的百分比
    percentage_table = contingency_table.div(contingency_table['總計'], axis=0).multiply(100).round(2)
    
    combined_table = contingency_table.astype(str) + ' (' + percentage_table.astype(str) + '%)'
    combined_table = combined_table.replace('nan (nan%)', '0 (0.00%)')
    
    return combined_table

def create_labeled_contingency_table(df, target_column, analysis_column):
    contingency_table = pd.crosstab(df[target_column], df[analysis_column], dropna=False)
    
    target_labels = contingency_table.index
    analysis_labels = contingency_table.columns
    
    new_target_labels = [f"{target_column}=缺失" if pd.isna(label) else f"{target_column}={label}" for label in target_labels]
    new_analysis_labels = [f"{analysis_column}=缺失" if pd.isna(label) else f"{analysis_column}={label}" for label in analysis_labels]
    
    contingency_table.index = new_target_labels
    contingency_table.columns = new_analysis_labels
    
    contingency_table = contingency_table.sort_index().sort_index(axis=1)
    
    return contingency_table

def chi_square_analysis(df):
    st.header("卡方檢驗和費雪精確檢驗分析")

    st.subheader("數據預覽")
    st.write(df.head())
    st.write("數據類型：")
    st.write(df.dtypes)

    target_column = st.selectbox("選擇目標列", df.columns)
    columns_to_analyze = st.multiselect("選擇要分析的列", [col for col in df.columns if col != target_column])

    # 添加連續變量閾值設置（包括目標列）
    thresholds = {}
    threshold_directions = {}
    if pd.api.types.is_numeric_dtype(df[target_column]):
        use_target_threshold = st.checkbox(f"為目標列 {target_column} 設置閾值?", key=f"threshold_target")
        if use_target_threshold:
            target_threshold = st.number_input(f"輸入 {target_column} 的閾值", value=float(df[target_column].mean()), key=f"threshold_value_target")
            target_direction = st.radio(f"選擇 {target_column} 的二分類方向", ["大於閾值為1", "小於等於閾值為1"], key=f"direction_target")
            thresholds[target_column] = target_threshold
            threshold_directions[target_column] = target_direction

    for column in columns_to_analyze:
        if pd.api.types.is_numeric_dtype(df[column]):
            use_threshold = st.checkbox(f"為 {column} 設置閾值?", key=f"threshold_{column}")
            if use_threshold:
                threshold = st.number_input(f"輸入 {column} 的閾值", value=float(df[column].mean()), key=f"threshold_value_{column}")
                direction = st.radio(f"選擇 {column} 的二分類方向", ["大於閾值為1", "小於等於閾值為1"], key=f"direction_{column}")
                thresholds[column] = threshold
                threshold_directions[column] = direction

    perform_fisher = st.checkbox("執行費雪精確檢驗（僅適用於2x2表格）", value=True)

    if st.button("開始分析"):
        if not columns_to_analyze:
            st.warning("請至少選擇一列進行分析。")
            return

        results = []
        # 如果目標列設置了閾值，則將其轉換為二分類
        if target_column in thresholds:
            if threshold_directions[target_column] == "大於閾值為1":
                df[f"{target_column}_binary"] = (df[target_column] > thresholds[target_column]).astype(int)
            else:
                df[f"{target_column}_binary"] = (df[target_column] <= thresholds[target_column]).astype(int)
            analysis_target = f"{target_column}_binary"
            st.write(f"已將目標列 {target_column} 轉換為二分類變量，閾值為 {thresholds[target_column]}，{threshold_directions[target_column]}")
        else:
            analysis_target = target_column

        for column in columns_to_analyze:
            st.subheader(f"分析：{target_column} vs {column}")
            
            try:
                # 如果設置了閾值，則將連續變量轉換為二分類
                if column in thresholds:
                    if threshold_directions[column] == "大於閾值為1":
                        df[f"{column}_binary"] = (df[column] > thresholds[column]).astype(int)
                    else:
                        df[f"{column}_binary"] = (df[column] <= thresholds[column]).astype(int)
                    analysis_column = f"{column}_binary"
                    st.write(f"已將 {column} 轉換為二分類變量，閾值為 {thresholds[column]}，{threshold_directions[column]}")
                else:
                    analysis_column = column


                contingency_table = create_labeled_contingency_table(df, analysis_target, analysis_column)
                st.write("列聯表：")
                st.write(add_totals_and_percentages(contingency_table))
                
                contingency_table_clean = contingency_table.drop(f"{analysis_target}=缺失", axis=0, errors='ignore')
                contingency_table_clean = contingency_table_clean.drop(f"{analysis_column}=缺失", axis=1, errors='ignore')
                
                if contingency_table_clean.empty or contingency_table_clean.shape[0] < 2 or contingency_table_clean.shape[1] < 2:
                    st.warning(f"移除缺失值後，{target_column} vs {column} 的數據不足以進行統計檢驗。")
                    continue

                chi2, p_value, dof, expected = chi2_contingency(contingency_table_clean)
                st.write(f"卡方檢驗 P 值：{p_value:.4f}")
                
                if p_value < 0.05:
                    st.write("🟢 變量之間存在統計學上顯著的關聯。")
                else:
                    st.write("🔴 變量之間不存在統計學上顯著的關聯。")
                
                result = {
                    '目標': target_column,
                    '分析列': column,
                    '卡方檢驗_p值': p_value
                }
                
                if perform_fisher and contingency_table_clean.shape == (2, 2):
                    oddsratio, fisher_p = fisher_exact(contingency_table_clean)
                    st.write(f"費雪精確檢驗 P 值：{fisher_p:.4f}")
                    result['費雪精確檢驗_p值'] = fisher_p
                elif perform_fisher:
                    st.info("費雪精確檢驗不適用（表格不是2x2）")
                    result['費雪精確檢驗_p值'] = '不適用'
                
                results.append(result)
            except Exception as e:
                st.error(f"分析 {target_column} vs {column} 時發生錯誤：{str(e)}")
            
            st.write("---")

        if results:
            results_df = pd.DataFrame(results)
            csv = results_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="下載分析結果",
                data=csv,
                file_name="分析結果.csv",
                mime="text/csv",
            )

# 添加新的函數來查詢列內資料
def column_data_query(df):
    st.header("所有列資料分佈查詢")

    st.subheader("數據預覽")
    st.dataframe(df.head())

    # 分析所有列
    for column in df.columns:
        st.subheader(f"{column} 列資料統計")
        
        # 計算每個值的數量和百分比
        value_counts = df[column].value_counts()
        value_percentages = df[column].value_counts(normalize=True) * 100
        value_stats = pd.DataFrame({
            '數量': value_counts,
            '百分比': value_percentages
        }).reset_index()
        value_stats.columns = [column, '數量', '百分比']
        value_stats['百分比'] = value_stats['百分比'].round(2)
        
        st.write(f"共有 {len(value_stats)} 個不同的值")
        st.dataframe(value_stats)

        # 檢查列的數據類型
        if pd.api.types.is_numeric_dtype(df[column]):
            # 數值型數據
            st.write(f"數據類型: 數值")
            st.write(df[column].describe())
            
            # 繪製直方圖
            fig = go.Figure(data=[go.Histogram(x=df[column])])
            fig.update_layout(title=f'{column} 數據分佈', xaxis_title=column, yaxis_title='頻率')
            st.plotly_chart(fig)
        else:
            # 分類型數據
            st.write(f"數據類型: 分類")

            # 繪製條形圖
            fig = go.Figure(data=[go.Bar(x=value_stats[column], y=value_stats['數量'], text=value_stats['百分比'].apply(lambda x: f'{x:.2f}%'), textposition='outside')])
            fig.update_layout(title=f'{column} 數據分佈', xaxis_title=column, yaxis_title='數量')
            st.plotly_chart(fig)

        # 提供下載選項
        csv = value_stats.to_csv(index=False)
        st.download_button(
            label=f"下載 {column} 查詢結果 CSV",
            data=csv,
            file_name=f"{column}_data_query.csv",
            mime="text/csv",
        )

        st.write("---")  # 添加分隔線

# 修改主應用函數
def main():
    st.set_page_config(page_title="實驗室數據分析", layout="wide")
    st.title("實驗室數據分析應用")

    # 在側邊欄中添加文件上傳功能
    with st.sidebar:
        uploaded_file = st.file_uploader("請上傳您的CSV文件", type="csv")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state['df'] = df
                st.success("文件上傳成功！")
                st.write(f"數據行數：{len(df)}")

                # 添加數據篩選功能
                do_filter = st.checkbox("是否進行數據篩選", value=False)
                if do_filter:
                    filter_column = st.selectbox("選擇用於篩選的列", df.columns)
                    is_numeric_column = pd.api.types.is_numeric_dtype(df[filter_column])
                    
                    if is_numeric_column:
                        filter_condition = st.selectbox("選擇篩���條件", ["等於", "大於", "小於"])
                        filter_value = st.number_input(f"輸入 {filter_column} 的篩選值", value=float(df[filter_column].mean()))
                    else:
                        filter_condition = "等於"
                        unique_values = df[filter_column].unique()
                        filter_value = st.selectbox(f"選擇 {filter_column} 的值進行篩選", unique_values)

                    # 應用篩選
                    if filter_condition == "等於":
                        filtered_df = df[df[filter_column] == filter_value]
                    elif filter_condition == "大於":
                        filtered_df = df[df[filter_column] > filter_value]
                    elif filter_condition == "小於":
                        filtered_df = df[df[filter_column] < filter_value]

                    st.session_state['df'] = filtered_df
                    st.write(f"篩選後的數據行數：{len(filtered_df)}")

    # 選擇分析類型
    analysis_type = st.sidebar.radio("選擇分析類型", [
        "列資料查詢",  
        "單一病歷號實驗室檢查項目趨勢圖", 
        "比較兩組連續變量統計差異", 
        "卡方檢驗和費雪精確檢驗分析",
        "Logistic 回歸分析",
        "生存分析"  # 新增的選項
    ])

    # 根據選擇的分析類型執行相應的函數
    if 'df' in st.session_state:
        if analysis_type == "比較兩組連續變量統計差異":
            continuous_variable_analysis(st.session_state['df'])
        elif analysis_type == "單一病歷號實驗室檢查項目趨勢圖":
            lab_test_trend(st.session_state['df'])
        elif analysis_type == "卡方檢驗和費雪精確檢驗分析":
            chi_square_analysis(st.session_state['df'])
        elif analysis_type == "Logistic 回歸分析":
            run_logistic_regression_analysis(st.session_state['df'])
        elif analysis_type == "列資料查詢":
            column_data_query(st.session_state['df'])
        elif analysis_type == "生存分析":
            run_survival_analysis(st.session_state['df'])
    else:
        st.info("請在側邊欄上傳一個CSV文件來開始分析。")

if __name__ == '__main__':
    main()

#cd /Users/mbpr/.cursor-tutor/projects/資料處理/
#streamlit run 實驗室數據分析入口.py