# Abstract

## English

Gradient boosting machines built on decision trees dominate tabular
prediction but expose their reasoning only through post-hoc, sampling-based
approximations such as SHAP or LIME. We study ZoneBoost, a gradient boosting
variant whose weak learner is built entirely from zone-grid descriptive
statistics -- quantiles, group counts, and empirical-Bayes-shrunk group
averages -- rather than trees, so that every prediction decomposes exactly
into inspectable main-effect and interaction terms with no sampling and no
approximation. We benchmark ZoneBoost against XGBoost, LightGBM, CatBoost,
and the Explainable Boosting Machine (its closest architectural relative)
across six public datasets spanning regression, binary and multiclass
classification, categorical-heavy features, and missing values, using
cross-validation and library-default hyperparameters throughout. A six-axis
ablation study isolates the effect of empirical-Bayes shrinkage, zone
boundary treatment, interaction order, categorical encoding, subsampling,
and boundary smoothing on predictive accuracy. Auxiliary experiments show
that a SHAP approximation of the same fitted model recovers ZoneBoost's
exact attributions only partially (mean per-row correlation $0.868$),
directly quantifying an approximation cost that is normally unmeasurable
because no ground truth exists for the black-box models SHAP is usually
applied to; that its conformalized quantile regression wrapper achieves
close-to-target coverage (0.88--0.90 against a 0.90 target) with
genuinely locally-adaptive interval width; that its optional classifier
calibration step did *not* improve Brier score or expected calibration
error on the classification dataset tested, a disclosed null result rather
than the expected direction; and that its drift-detection utility
correctly surfaces the largest feature-importance shifts under a real,
non-synthetic covariate shift. Relative to the best tree-based competitor
per dataset, ZoneBoost's regression RMSE ranges from 3\% better (Diabetes,
where it beats every tree-based model tested) to 21\% worse (Bike Sharing
Demand, its largest gap); on classification it is the single best model
tested on Wine (98.3\% accuracy vs.\ 97.2\% for the next-best model) and
within 0.7--1.2 percentage points of the best tree-based model on the
other two classification datasets. A six-axis ablation study designed to
test our working hypothesis -- that empirical-Bayes shrinkage explains
ZoneBoost's small-sample competitiveness -- refutes it: near-zero shrinkage
performs statistically indistinguishably from the default across every
dataset tested. The actual largest-effect axis, by a wide margin, is the
adaptive, per-round zone-boundary search, whose removal costs up to 56\%
relative RMSE and 3 percentage points of classification accuracy, more
than any other architectural component tested. These results position
ZoneBoost as a viable choice when exact, auditable attribution is a hard
requirement and the data is small-to-moderate in size or multiclass, at a
quantified and dataset-dependent cost in raw predictive accuracy relative
to opaque tree-based boosting.

**Keywords:** gradient boosting, interpretable machine learning, explainable
boosting machine, ablation study, conformal prediction, tabular data

## 中文摘要（繁體中文）

以決策樹為弱學習器的梯度提升機（gradient boosting machine）主宰了表格資料的
預測任務，但其決策過程只能透過 SHAP、LIME 等事後、基於取樣近似的方法來解釋。
本文研究 ZoneBoost——一種以「區域格統計量」（zone-grid descriptive
statistics，包括分位數、群組筆數與經驗貝氏收縮後的群組平均）取代決策樹作為弱
學習器的梯度提升變體，使每一筆預測都能精確分解為可直接檢視的主效應與交互作用
項，不需取樣、亦無近似誤差。我們在六個公開資料集（涵蓋迴歸、二元與多類別分
類、高類別特徵、缺失值情境）上，以交叉驗證與各套件預設超參數，將 ZoneBoost 與
XGBoost、LightGBM、CatBoost 以及其架構上最接近的可解釋模型 Explainable
Boosting Machine（EBM）進行比較。六個維度的消融研究（ablation study）分離出經
驗貝氏收縮、區域邊界處理方式、交互作用階數、類別變數編碼方式、子抽樣、邊界平
滑化等模組對預測準確度的個別影響。附加實驗顯示：同一模型的 SHAP 近似值僅能部
分還原 ZoneBoost 的精確歸因（每列平均相關係數為 0.868），直接量化出一項通常無
法測量的近似代價——因為 SHAP 平常應用的黑箱模型並無精確答案可供比對；其保形化
分位數迴歸（conformalized quantile regression）包裝器達成接近目標的涵蓋率
（實測 0.88–0.90，目標為 0.90）且區間寬度確實隨輸入而調整；其分類器校準選項在
受測資料集上並未降低 Brier 分數或預期校準誤差，此為誠實揭露的空結果，而非預期
方向；以及其飄移偵測工具能在真實（非人工模擬）共變量飄移下正確標示出重要性變
化最大的特徵。與各資料集中表現最佳的樹狀提升模型相比，ZoneBoost 的迴歸均方根
誤差（RMSE）介於優於 3%（Diabetes，此資料集上 ZoneBoost 勝過所有受測樹狀模
型）至劣於 21%（Bike Sharing Demand，差距最大之情形）之間；在分類任務中，
ZoneBoost 在 Wine 資料集上是所有受測模型中準確度最高者（98.3%，次佳模型為
97.2%），在另外兩個分類資料集上則與最佳樹狀模型相差 0.7 至 1.2 個百分點以內。
本文原先的工作假說——經驗貝氏收縮是 ZoneBoost 在小樣本情境下具競爭力的關鍵機
制——經六維度消融研究檢驗後遭到推翻：在所有受測資料集上，近乎零收縮與預設收縮
強度的表現在統計上並無可辨識差異。真正影響最大的模組，且效果遠超其他模組，是
逐輪次重新搜尋的自適應區域邊界劃分機制，移除後迴歸均方根誤差最高可上升 56%、
分類準確度最高可下降 3 個百分點，效果大於本研究中測試的任何其他架構模組。這些
結果顯示，當精確且可稽核的歸因是硬性需求、且資料規模為中小型或屬多類別分類問
題時，ZoneBoost 是可行的選擇，其相對於不透明樹狀提升模型的預測準確度代價則因
資料集而異，但可被量化。

**關鍵詞：** 梯度提升、可解釋機器學習、可解釋提升機、消融研究、保形預測、表格
資料
