# 公开数据接入（已确认）

## 来源与固定版本

采用数据库维护者的[官方数据归档仓库](https://github.com/Jesperkemist/perovskitedatabase_data)，固定 commit 为 `9b6ed4ca72910cbdf27b9832cf0c081caa936af3`。该仓库 README 标明数据采用 CC BY 4.0，研究使用时应引用所用归档版本及[数据库论文](https://doi.org/10.1038/s41560-021-00941-3)。当前固定的是 Git commit，正式论文阶段还需核对对应 Zenodo 归档版本及引用信息。

CSV 文件名：`Perovskite_database_content_all_data.csv`。

- 文件大小：84,659,951 字节。
- SHA256：`e722ded60e6162a47b00746c6215498dcf0ce9dd004703fd01ab560dfc213ba2`。
- 数据本体通过 Git LFS media 地址下载；普通 raw 地址只提供 133 字节的指针文件。
- 原始文件位于 `data/raw/`，由 Git 忽略，可用下载脚本重建。

## 已执行的检查

已下载数据本体，并校验文件大小及 SHA256。检查脚本逐条读取全部记录，未发现字段数量不一致或重复表头。

- 记录数：42,443。
- 字段数：410。
- 不同 DOI 数：7,358。DOI 只做去首尾空格和转小写，尚未做 URL 前缀等进一步规范化。
- `JV_default_PCE` 可解析且处于 0–100 范围的记录：41,519，均有非空 DOI。
- 器件架构：nip 29,627 条，pin 12,764 条；其余为其他或未知架构。

这些是固定版本的实际统计，不代表当前在线数据库的总量。全部字段名、缺失比例及示例见 [public_data_profile.json](./metadata/public_data_profile.json)。

## 已确认的候选字段

- 分组标识：`Ref_DOI_number`，完整记录 42,443 条。只用于分组和来源追溯，不作为预测特征。
- 预测标签：`JV_default_PCE`，缺失约 2.18%。
- 测量口径：`JV_default_PCE_scan_direction`。包含 Reversed、Forward、Stabilised，下一步需要统一标签口径。
- 器件背景：`Cell_architecture`、`Cell_area_measured`。后者缺失约 1.68%；`Cell_area_total` 缺失约 97.74%，不能与测量面积混用。
- 组分：`Perovskite_composition_short_form`，非空率 100%，仍需检查表达是否足够完整。后续应结合组成离子与比例字段，不能仅靠短名称确定精确配方。
- 制备方式：`Perovskite_deposition_procedure`，缺失约 0.34%。
- 溶剂：`Perovskite_deposition_solvents`、`Perovskite_deposition_solvents_mixing_ratios`，缺失约 3.71% 和 5.00%。
- 退火：`Perovskite_deposition_thermal_annealing_temperature`、`Perovskite_deposition_thermal_annealing_time`，缺失约 8.08% 和 10.44%。
- 沉积湿度：`Perovskite_deposition_synthesis_atmosphere_relative_humidity`，缺失约 98.39%。
- 反应溶液浓度：`Perovskite_deposition_reaction_solutions_concentrations`，缺失约 96.57%。

## 对下一步的影响

已选择 nip、单步旋涂、单阶段退火的铅基小面积器件子集，保留 9,913 条记录并完成首轮训练。实际清洗规则与单位见[字段字典](../docs/field_dictionary.md)，训练结果见项目 README。

温湿度、浓度不能直接按原需求作为首版必备输入。即使个别字段整体覆盖率较高，也必须检查这些字段在同一批记录中的共同完整程度。

工艺字段包含 `;`、`>>`、`|` 等结构符号。例如退火温度有 `70; 100`，制备方式有 `Co-evaporation >> Evaporation >> Spin-coating`。已查阅官方字段说明：这些符号区分混合/程序阶段、制备步骤与器件层。首版保守排除多阶段、多层和多步工艺。官方溶剂比例优先为体积比，但允许其他口径，因此本模型只称其为记录比例份额。

当前缺失统计仅识别整格的空值及常见未知标记。像 `nan >> 1 mg/ml` 这样的部分缺失字符串仍算非空，因此非空率不等于可用于训练的比例。

PCE 0–100 仅用于检查能否解析，尚未制定训练筛选阈值。0% 的失败实验不会在这一阶段被删除。检查未修改原始数据。

## 复现命令

在项目根目录使用 PowerShell：

```powershell
./scripts/download_data.ps1
python scripts/inspect_data.py data/raw/Perovskite_database_content_all_data.csv --output data/metadata/public_data_profile.json
```

检查脚本仅依赖 Python 标准库。本机 `py` 启动器指向的 Python 未能启动，本次实际使用 Codex 内置运行时完成检查：

```powershell
& 'C:\Users\yz130\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/inspect_data.py data/raw/Perovskite_database_content_all_data.csv --output data/metadata/public_data_profile.json
```

下载脚本遇到不同版本的已有 CSV 会停止，避免覆盖；中断产生的 `.download` 文件也需检查后处理。

## 后续进度

本步已由用户确认。数据清洗、训练、推理和研究版工艺探索已实现；组内历史数据、实验验证和物理描述符来源仍待接入。
