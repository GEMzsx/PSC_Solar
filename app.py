"""Local Streamlit research tool backed exclusively by the trained model."""
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

from psc_solar.model import load_bundle, predict
from psc_solar.optimize import explore
from psc_solar.schema import FEATURES

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts' / 'baseline'
st.set_page_config(page_title='PSC Solar 工艺预测', layout='wide')


@st.cache_resource
def get_bundle(path, modified_time):
    return load_bundle(path)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')


st.title('PSC Solar 工艺预测')
st.caption('公开历史数据研究版 · nip / 单步旋涂 / 铅基小面积器件 / 反向扫描 PCE')
model_path = ARTIFACTS / 'model.joblib'
if not model_path.exists():
    st.info('请先在项目根目录运行 python -m psc_solar train，再刷新页面。')
    st.stop()
bundle = get_bundle(str(model_path), model_path.stat().st_mtime_ns)
report = bundle['metadata']
st.warning('当前模型未经组内实验验证，不能用于大面积器件外推。所有输入默认采用 AM 1.5、100 mW/cm² 测量口径。')
overview, inference, optimization, records = st.tabs(['训练结果', '性能预测', '工艺探索与实验建议', '实验记录'])

with overview:
    cols = st.columns(4)
    cols[0].metric('筛选后样本', f"{report['retained_rows']:,}")
    cols[1].metric('测试 MAE', f"{report['test']['mae_percentage_points']:.2f} 个百分点")
    cols[2].metric('测试 R²', f"{report['test']['r2']:.3f}")
    cols[3].metric('区间实测行覆盖率', f"{report['test']['row_interval_coverage']:.1%}")
    st.write(f"选择模型：{report['selected_model']}。均值基线测试 MAE：{report['test_mean_baseline']['mae_percentage_points']:.2f} 个百分点。")
    st.write(f"区间名义目标为新论文分组 90% 覆盖；测试分组全部样本覆盖率为 {report['test']['independent_group_all_rows_coverage']:.1%}，固定区间宽度为 {report['test']['interval_width_percentage_points']:.2f} 个百分点。")
    st.caption('论文分组分离不等于实验室分离。覆盖结论依赖分组可交换性；区间可能较宽，也不代表特定配方成功概率。')
    left, right = st.columns(2)
    with left:
        st.subheader('独立测试集预测')
        test = pd.read_csv(ARTIFACTS / 'test_predictions.csv')
        st.scatter_chart(test, x='pce', y='prediction', x_label='实测 PCE (%)', y_label='预测 PCE (%)')
    with right:
        st.subheader('验证集因素分析')
        importance = pd.read_csv(ARTIFACTS / 'feature_importance.csv')
        chart = alt.Chart(importance).mark_bar().encode(
            x=alt.X('mae_increase_mean:Q', title='打乱字段组后的 MAE 增量（百分点）'),
            y=alt.Y('feature:N', title='输入字段组', sort='-x'),
            tooltip=['feature', 'mae_increase_mean', 'mae_increase_std'],
        ).properties(height=340)
        st.altair_chart(chart, width='stretch')
        st.caption('预测关联，不是因果关系。A 位、X 位和溶剂的份额分别整体打乱，以保留和为 1 的约束；与其他工艺的关联仍可能被打乱。')
    st.subheader('数据划分')
    st.dataframe(pd.DataFrame(report['splits']).T, width='stretch')
    small_sample_path = ARTIFACTS / 'small_sample_results.csv'
    if small_sample_path.exists():
        st.subheader('50–200 条样本预演')
        small = pd.read_csv(small_sample_path)
        summary = small.groupby(['training_rows', 'model']).mae_percentage_points.agg(['mean', 'std']).reset_index()
        st.dataframe(summary.rename(columns={'training_rows': '训练条数', 'model': '模型', 'mean': '验证 MAE 均值（百分点）', 'std': '三次采样标准差'}), width='stretch')
        st.caption('三次按论文顺序抽样，最后一篇可部分取样；评价使用开发验证集，未使用测试集。这是公开数据小样本预演，不代表组内数据精度。')
    with st.expander('查看模型范围与限制'):
        st.json(report['scope'])
        st.write('\n'.join(report['limitations']))
    st.download_button('下载训练评价 JSON', json.dumps(report, ensure_ascii=False, indent=2), 'metrics.json', 'application/json')

example = pd.read_csv(ARTIFACTS / 'example_input.csv', keep_default_na=False).iloc[0].to_dict()
with inference:
    st.subheader('输入组分与工艺')
    st.caption('A 位与 X 位为归一化占位份额；B 位固定 Pb。溶剂输入为记录比例份数，并按总和归一化。')
    with st.form('recipe_form'):
        a, b, c = st.columns(3)
        with a:
            cs = st.number_input('Cs 份额 (%)', min_value=0., max_value=100., value=float(example['a_cs']) * 100, key='a_cs')
            fa = st.number_input('FA 份额 (%)', min_value=0., max_value=100., value=float(example['a_fa']) * 100, key='a_fa')
            st.caption('MA 份额 = 100% − Cs − FA')
            br = st.number_input('Br 份额 (%)', min_value=0., max_value=100., value=float(example['x_br']) * 100)
            temperature = st.number_input('退火温度 (°C)', min_value=-10., max_value=400., value=float(example['anneal_temperature_c']))
            time = st.number_input('退火时间 (min)', min_value=0., max_value=1440., value=float(example['anneal_time_min']))
            area = st.number_input('测量面积 (cm²)', min_value=.000001, max_value=1., value=float(example['area_cm2']), format='%.6f')
        with b:
            solvent_parts = {name: st.number_input(f'{name.upper()} 比例份数', min_value=0., max_value=1000., value=float(example['solvent_' + name]), key='parts_' + name) for name in ['dmf', 'dmso', 'gbl', 'nmp']}
        with c:
            categories = {}
            for key, label in [('etl', '电子传输层 ETL'), ('htl', '空穴传输层 HTL'), ('quenching_media', '反溶剂 / 淬火介质'), ('additives', '添加剂（Unreported 为未报告）')]:
                options = report['feature_categories'][key]
                categories[key] = st.selectbox(label, options, index=options.index(example[key]))
        submitted = st.form_submit_button('预测 PCE', type='primary')
    if submitted:
        for key in ['recipe', 'prediction', 'exploration']:
            st.session_state.pop(key, None)
        try:
            if cs + fa > 100:
                raise ValueError('Cs 和 FA 份额之和不能超过 100%。')
            total = sum(solvent_parts.values())
            if total <= 0:
                raise ValueError('溶剂比例份数之和必须大于零。')
            recipe = {'a_cs': cs / 100, 'a_fa': fa / 100, 'a_ma': (100 - cs - fa) / 100,
                      'x_br': br / 100, 'x_i': 1 - br / 100,
                      **{'solvent_' + name: value / total for name, value in solvent_parts.items()},
                      'anneal_temperature_c': temperature, 'anneal_time_min': time, 'area_cm2': area, **categories}
            result = predict(bundle, pd.DataFrame([recipe])).iloc[0]
            st.session_state['recipe'] = recipe
            st.session_state['prediction'] = result.to_dict()
            st.session_state.pop('exploration', None)
        except ValueError as error:
            st.error(str(error))
    if 'prediction' in st.session_state:
        result = st.session_state['prediction']
        st.metric('预测 PCE', f"{result['pce_prediction']:.2f}%")
        st.write(f"90% 名义分组覆盖区间：{result['pce_lower']:.2f}% ～ {result['pce_upper']:.2f}%")
        if result['warnings']:
            st.warning('适用范围提醒：' + result['warnings'])
        st.caption('区间不裁剪至物理范围。负下限表示不确定性较大。输入即使落在各字段范围内，也可能是训练中未见过的组合。')
        st.caption('这里展示最近一次成功提交的预测；修改表单后请再次点击“预测 PCE”。')
        st.download_button('下载本次输入', csv_bytes(pd.DataFrame([st.session_state['recipe']])), 'recipe.csv', 'text/csv')
        output = pd.DataFrame([{**st.session_state['recipe'], **result}])
        st.download_button('下载本次预测结果', csv_bytes(output), 'prediction.csv', 'text/csv')
    st.divider()
    st.subheader('CSV 批量预测')
    st.download_button('下载输入格式示例', (ARTIFACTS / 'example_input.csv').read_bytes(), 'example_input.csv', 'text/csv')
    upload = st.file_uploader('上传标准化输入 CSV（最多 1,000 行）', type=['csv'])
    if upload is not None:
        try:
            frame = pd.read_csv(io.BytesIO(upload.getvalue()), keep_default_na=False)
            if len(frame) > 1000:
                raise ValueError('最多允许 1,000 行。')
            results = predict(bundle, frame)
            output = pd.concat([frame.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
            st.dataframe(output, width='stretch')
            st.download_button('下载批量预测结果', csv_bytes(output), 'predictions.csv', 'text/csv')
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))

with optimization:
    st.subheader('固定配方，探索退火温度与时间')
    st.write('先在“性能预测”中提交配方。这里保持组分、溶剂、器件面积和传输层不变，仅搜索退火参数。')
    st.caption('采用与当前配方相近的训练记录限制搜索支持范围。温度扰动使用绝对 °C，时间扰动使用百分比。结果仅用于实验排期讨论。')
    if 'recipe' in st.session_state:
        recipe = st.session_state['recipe']
        limits_t = report['feature_ranges']['anneal_temperature_c']
        limits_time = report['feature_ranges']['anneal_time_min']
        with st.form('optimization_form'):
            col_t, col_time, col_target = st.columns(3)
            with col_t:
                lower_t = st.number_input('最低温度 (°C)', value=float(max(limits_t['min'], recipe['anneal_temperature_c'] - 15)))
                upper_t = st.number_input('最高温度 (°C)', value=float(min(limits_t['max'], recipe['anneal_temperature_c'] + 15)))
                tolerance_t = st.number_input('温度波动 ±°C', min_value=0., max_value=30., value=5.)
            with col_time:
                lower_time = st.number_input('最短时间 (min)', value=float(max(limits_time['min'], recipe['anneal_time_min'] * .7)))
                upper_time = st.number_input('最长时间 (min)', value=float(min(limits_time['max'], max(recipe['anneal_time_min'] * 1.3, lower_time + 1))))
                tolerance_time = st.number_input('时间波动 ±%', min_value=0., max_value=50., value=10.)
            with col_target:
                target = st.number_input('目标 PCE (%)', min_value=0., max_value=100., value=15.)
                batch = st.number_input('下一批实验点数', min_value=1, max_value=20, value=6)
            run = st.form_submit_button('搜索工艺候选点')
        if run:
            st.session_state.pop('exploration', None)
            try:
                with st.spinner('计算候选点与扰动响应…'):
                    st.session_state['exploration'] = explore(bundle, recipe, (lower_t, upper_t), (lower_time, upper_time), target, tolerance_t, tolerance_time / 100, batch_size=int(batch))
            except ValueError as error:
                st.error(str(error))
        if 'exploration' in st.session_state:
            exploration = st.session_state['exploration']
            grid, suggested = exploration['grid'], exploration['recommendations']
            st.write(f"相近训练记录：{exploration['comparable_training_records']}。有支持的网格点：{int(grid.supported.sum())}。保守下界达到目标的点：{int(grid.model_passes_target.sum())}。")
            if not grid.model_passes_target.any():
                st.info('当前范围内未找到保守下界达到目标的窗口；这可能由区间较宽或局部数据不足造成。')
            st.dataframe(grid[['anneal_temperature_c', 'anneal_time_min', 'pce_prediction', 'conservative_lower', 'supported', 'model_passes_target']], width='stretch')
            st.download_button('下载完整工艺网格', csv_bytes(grid), 'process_grid.csv', 'text/csv')
            if not suggested.empty:
                st.subheader('下一批实验候选')
                st.dataframe(suggested[['anneal_temperature_c', 'anneal_time_min', 'pce_prediction', 'pce_lower', 'pce_upper', 'recommendation_reason']], width='stretch')
                st.download_button('下载实验建议（含完整预测输入）', csv_bytes(suggested), 'experiment_suggestions.csv', 'text/csv')
            else:
                st.info('未找到满足支持范围且与已测点有足够距离的新实验候选。')
            st.caption('performance 表示偏向性能；exploration_proxy 表示偏向模型分歧或新颖性。不是已经验证的信息增益或实验结论。')

with records:
    st.subheader('标准化实验记录')
    st.write('每个测量器件填写一行；同批次重复器件共享 batch_id，sample_id 应唯一。缺测留空，失败实验保留并填写原因。')
    st.download_button('下载实验记录模板', (ROOT / 'templates' / 'experiment_record.csv').read_bytes(), 'experiment_record.csv', 'text/csv')
    st.write('组内数据、薄膜表征和溶剂物理描述符仍需核实来源与口径。当前模板是采集接口，首版模型只使用公开数据中的组分和工艺字段。')
    st.code('python -m psc_solar ingest --input 填写后的实验记录.csv', language='powershell')
    st.caption('导入会验证样本编号、批次、状态、单位，跳过完全相同的重复样本，遇到冲突停止。记录存入本地 data/lab；不符合当前模型范围的实验仍保留并标明原因，不自动重新训练。status 使用 success、failed 或 unmeasured。')
    st.markdown((ROOT / 'docs' / 'field_dictionary.md').read_text(encoding='utf-8'))
