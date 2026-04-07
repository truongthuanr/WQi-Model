Reviewer 1: Thank you for the opportunity to review this manuscript. The study addresses a relevant problem in intensive aquaculture: the need for faster and operationally useful tools to anticipate alkalinity dynamics in shrimp farming systems, where carbonate buffering and pH stability directly affect water quality management and production performance. The manuscript compares three machine-learning approachesâ€”Random Forest Regression (RFR), Support Vector Regression (SVR), and Artificial Neural Networks (ANN)â€”using a dataset of 4,716 records collected in Vietnam, together with an external validation set of 527 additional samples from other sites. The main contribution of the paper lies in treating alkalinity as an explicit forecasting target, supported by a broader empirical basis than that used in many previous aquaculture studies. Overall, I find the topic relevant and the dataset potentially valuable, but the current version still contains methodological and traceability gaps that prevent a sufficiently rigorous audit of how the final model was built, compared, and validated.

From a technical standpoint, the manuscript attempts to reduce reliance on conventional monitoring based on manual sampling and laboratory testing by proposing a predictive approach intended to support farm decisions. To do so, the authors use data collected four times per day over an eleven-month period, incorporate physicochemical and farming-management variables, apply z-score normalization, compare models with RÂ² and RMSE, and extend the assessment to an external dataset. They also add a seasonal characterization based on WQI, ANOVA, and FDR correction. The central result is that RFR outperforms SVR and ANN for both same-day prediction and next-day forecasting, although the manuscript itself acknowledges a loss of accuracy at alkalinity extremes. The general study architecture is understandable; however, the scientific strength of the paper depends on several clarifications that are not yet sufficiently documented.

Among the manuscript's genuine strengths, I would first highlight the relevance of the research problem. The paper does not simply repeat generic AI applications to water quality, but instead explains why alkalinity deserves dedicated treatment in intensive shrimp systems, as a core component of buffering capacity and chemical stability. Second, the empirical effort is substantial: the 4,716-record dataset with four daily measurements across eleven months, combined with 527 external samples, clearly exceeds the scale of studies based on only a few ponds or short pilot trials. Third, it is valuable that the authors attempt to move beyond internal validation and test transferability with data from other regions, thereby increasing the practical relevance of the work. Finally, I appreciate that the manuscript explicitly acknowledges reduced predictive performance at alkalinity extremes, which avoids presenting the algorithm as a universally reliable solution and leads to a more credible discussion of its limits.

My first critical concern relates to computational reproducibility.

--- Response ---

Thank you for this important point. We have fully reconstructed the computational workflow and now provide a stepwise, reproducible pipeline. The revised manuscript explicitly reports: (i) which predictors were normalized, (ii) where and how scaling was fit to prevent leakage, (iii) the exact split sequence, (iv) the time-blocked nested cross-validation setup (including folds), (v) model-specific hyperparameter search spaces, and (vi) the final model-selection rule. We also added a dedicated workflow table/appendix to support independent auditability. The methods mention z-score normalization, a 60/20/20 split, "auto mode with 200 loops," GridSearchCV, and a "nested, time-blocked" cross-validation scheme, but the manuscript does not specify which variables were normalized, whether scaling was fit only on training data or on the full dataset, which hyperparameters were explored for each model, how many folds were used, or how the time-blocked procedure was reconciled with a previous random split. This is not a minor issue. With data collected four times per day over eleven months, the partitioning and scaling strategy directly affects the risks of information leakage and inflated model performance. Without that level of traceability, an independent reader cannot reproduce or audit the workflow. I strongly recommend that the authors reconstruct the analytical pipeline explicitly, ideally through a dedicated methodological table or appendix covering preprocessing, splitting sequence, hyperparameter tuning, and final model selection criteria.

My second critical point concerns the Water Quality Index.

--- Response ---

We agree and have added the full WQI definition. The revised text now includes the equation, component variables, normalization procedure, weighting strategy, interpretive scale, and methodological references. We also clarify the analytical role of WQI (seasonal characterization rather than the prediction target). The manuscript states that a composite WQI was calculated from normalized physicochemical and nutrient parameters and then used for seasonal comparison, yet the formula, weighting scheme, interpretive scale, and methodological justification are not provided. As a result, the reported seasonal differences in WQI are not fully auditable. Since the paper introduces WQI as an additional analytical layer and uses it to support claims about structured environmental variability, the index must be operationally defined. The necessary improvement is straightforward: include the equation, component variables, normalization procedure, weighting strategy, and source methodology for the index.

A third critical issue is an internal inconsistency between variable selection, feature-importance interpretation, and the specification of the final model.

--- Response ---

Thank you for identifying this inconsistency. We have rewritten this section to clearly separate: (i) feature relevance analysis, (ii) reduced-input scenarios, and (iii) final predictor sets used for each experiment. We also added an experimental-design table mapping each reported result (including Table 3) to its exact predictor set and selection criterion. In the section discussing variable weights, the manuscript states that shrimp age, temperature, pH, salinity, water level, farming method, pond type, season, area, and transparency were identified as relevant inputs; immediately afterward, however, it states that to build an "easy and cheap" model these indicators were suggested for elimination from the input sources, while farming technologies and pond type were used for model evaluation. This sequence is contradictory and leaves unresolved which exact predictor set was used in the models reported in Table 3. Since that table supports the main comparison among ANN, RFR, and SVR, this ambiguity undermines the interpretability of the paper's central result. I recommend a complete rewrite of this section, supported by an experimental-design table showing, at each stage, which predictor set was used, under what selection criterion, and with what corresponding output.

A fourth critical concern affects the external validation.

--- Response ---

We have expanded external validation reporting and now provide site-specific quantitative metrics. For each site, we report n, alkalinity range, mean, standard deviation, R2, RMSE, MAE, and bias, together with a short discussion of distributional differences (including extreme-value representation) and implications for transferability. The manuscript reports that RFR performance was tested using 527 samples from the original site and two additional farming regions, and Figure 5 presents site-wise correlations. However, no site-specific numerical performance metrics are reported, nor are sample sizes per site, alkalinity ranges, analytical comparability, or potential differences in the distribution of extreme cases. In its present form, the statement that the model retains robust predictive ability on independent datasets is stronger than the evidence actually shown. If cross-site generalization is one of the manuscript's main claims, then the external validation must be documented with the same rigor as the internal evaluation. I recommend adding a dedicated validation table reporting n, range, mean, standard deviation, RÂ², RMSE, MAE, and bias for each site.

My fifth critical concern relates to proportionality between results and conclusions.

--- Response ---

We agree and have moderated the framing of claims in the Abstract and Discussion. Statements about reduced laboratory dependence, operational efficiency, and SDG relevance are now presented as potential applications rather than demonstrated outcomes. The revised conclusions are explicitly limited to predictive-performance evidence supported by this study. In both the abstract and the discussion, the manuscript argues that the proposed framework can reduce reliance on repeated laboratory testing, strengthen water-quality surveillance, and function as a sustainability-oriented tool linked to SDG 6, 12, and 14. However, the study does not quantify cost savings, actual reductions in sampling effort, improved response times, environmental gains, or operational benefits relative to conventional monitoring. What is demonstrated is comparative predictive performance using RÂ² and RMSE, not sustainability performance or field-level efficiency. This does not negate the applied interest of the study, but it does require a more restrained framing of the conclusions. I suggest reformulating those claims as plausible applications rather than as outcomes demonstrated by the current design.

Among the major comments, the first is the mismatch between the temporal structure of the data and the evaluation strategy.

--- Response ---

This point has been addressed by clarifying the temporal evaluation design in detail. We now state whether model selection and testing were chronological, random, or hybrid, and explain the rationale. We also added an explicit discussion of optimism risk when temporal separation is not strictly enforced in high-frequency series. The study relies on a high-frequency series with likely autocorrelation, yet the methods report a random dataset split. Although "time-blocked" cross-validation is mentioned later, the manuscript does not explain how the two decisions were combined or which one actually governed model selection. This ambiguity weakens confidence in the reported RÂ² and RMSE values, because randomization in time-series data can inflate performance by mixing nearby observations across training and testing sets. The authors should clarify whether the main evaluation was chronological, random, or hybrid, and explicitly discuss the risk of optimism if strict temporal separation was not used.

The second major comment concerns documentation of measurement workflow and quality control.

--- Response ---

We have added a dedicated QA/QC subsection describing source synchronization (IoT, laboratory, Secchi, and FMS records), sampling-to-analysis timing, calibration and analytical checks, and treatment of uncertainty in the alkalinity target variable. The manuscript states that some variables were obtained through IoT sensors, others through laboratory spectrophotometry, water transparency through Secchi disk, and farming-management information through an FMS. Yet it remains unclear how these sources were synchronized in time, how much time elapsed between sample collection and alkalinity determination, whether duplicates, blanks, calibrations, or analytical checks were used, and what uncertainty was associated with the target variable itself. In a paper whose value depends on forecasting alkalinity from operational data, the quality and consistency of the target measurement deserve clearer documentation. A concise QA/QC subsection would significantly strengthen the paper.

The third major comment is that model evaluation is somewhat narrow for an applied environmental forecasting problem.

--- Response ---

We expanded the performance audit beyond R2 and RMSE. The revised manuscript now includes MAE, mean bias, and residual diagnostics stratified by alkalinity range/quantiles, with focused analysis of predictive behavior at low and high alkalinity extremes. The paper compares models using RÂ² and RMSE, yet it explicitly acknowledges that errors increase at alkalinity extremes. Under these conditions, additional metrics such as MAE, mean bias, range-stratified performance, or residual analysis by quantiles would be highly informative, especially because practical utility depends not only on average fit but also on performance under critical conditions. I recommend broadening the performance audit with complementary metrics and a dedicated analysis of errors at the extremes.

The fourth major comment concerns interpretation of variable importance.

--- Response ---

We revised this section to avoid direct comparability claims where methods differ. The manuscript now explains how importance was derived for each model and explicitly cautions against one-to-one interpretation across SVR and RFR importance measures unless methodologically aligned. Table 2 reports "weight values" for SVR and RFR, and the text then compares both sets as though they were directly equivalent. However, the manuscript does not explain how importance was derived in each case or whether the two measures are methodologically comparable. This matters because the discussion uses these rankings to support explanatory claims regarding shrimp age, pH, TDS, and salinity. If feature importance was not obtained through conceptually equivalent procedures, the comparison should be presented more cautiously and the extraction method for each model should be made explicit.

The fifth major comment relates to the discussion section.

--- Response ---

We have strengthened the Discussion by linking dominant predictors to carbonate chemistry and farming-system dynamics. In particular, we now interpret shrimp age as a proxy for evolving biogeochemical and management conditions across production stages, rather than as a purely statistical rank. The manuscript correctly reiterates that RFR performed best, but it devotes less effort to explaining why certain variables dominate from the perspective of carbonate chemistry, farming dynamics, or differences between post-larval and grow-out ponds. It would also be valuable to interpret shrimp age more deeply as a proxy for changing biogeochemical and management conditions rather than treating it merely as a ranked feature. Such expansion would increase the scientific value of the paper by linking algorithmic output to ecological and operational mechanisms.

As for minor comments, Figure 2 requires immediate correction.

--- Response ---

Corrected. The template artifact in Figure 2 has been removed, and the figure content has been aligned with the actual model-evaluation workflow. The "MODEL EVALUATION" block includes the phrase "Read resources to support your hypothesis," which is clearly unrelated to a model-evaluation workflow and appears to be a template artifact or unedited placeholder. Although this does not invalidate the analysis, it does affect editorial credibility and should be corrected before the manuscript can be considered further.

I also recommend a careful revision of the scientific English throughout the manuscript.

--- Response ---

We performed comprehensive language editing to improve scientific clarity, technical precision, and grammar throughout the manuscript, including correction of ambiguous model-description phrases. Several formulations are imprecise or grammatically weak and, in some cases, obscure the technical meaning. Expressions such as "random neurons each," "most recommended water model," or the ambiguous wording around variable elimination need scientific language editing, not merely cosmetic polishing.

I believe the manuscript addresses a relevant topic, draws on a valuable dataset, and has clear applied potential, but it does not yet reach a sufficient level of methodological solidity for acceptance in its current form. The main issue is not the research question itself, nor the relevance of the case study, but rather the lack of transparency in key analytical decisions and the breadth of certain claims relative to the evidence actually presented. My editorial recommendation is major revisions. The study could be strengthened substantially if the authors reconstruct the computational pipeline with precision, define the WQI formally, clarify variable selection, document external validation more rigorously, and moderate the applied conclusions so that they remain strictly proportional to what was demonstrated.

Priority actions should therefore include: (1) a reproducible description of preprocessing, temporal or random splitting, hyperparameter tuning, and final model selection; (2) full definition of the WQI; (3) resolution of the inconsistency between feature-importance interpretation and the predictor sets used in Table 3; (4) expanded external validation with site-specific metrics; (5) complementary performance metrics and a focused analysis of extreme-value errors; and (6) correction of figures and technical language before resubmission.


Reviewer 2: The manuscript entitled "Machine learning-enabled alkalinity forecasting for resource-efficient and sustainable water-quality monitoring in managed aquatic systems" presents a study on the application of machine learning (ML) models Random Forest Regression (RFR), Support Vector Regression (SVR), and Artificial Neural Networks (ANN) to forecast alkalinity in intensive shrimp pond aquaculture systems. The following suggestions are follows
1. Introduction section is week especially authors should mention why they have chosen the specific ML models such as RFR, SVM and ANN for their study.
--- Response ---
Thank you. We expanded the Introduction to justify model selection (RFR, SVR, and ANN) based on their suitability for nonlinear behavior, mixed predictor types, and prior performance in environmental forecasting literature.
2. Also, the research gap is not significant up to the level. Authors are suggested to enhance this portion a little bit.
--- Response ---
We revised the research-gap statement to clearly position the novelty of this work: alkalinity as an explicit forecasting target using high-frequency operational data with external cross-site validation.
3. Authors need to highlight their research objectives clearly with bullet points
--- Response ---
Implemented. We now present explicit research objectives in bullet-point form at the end of the Introduction.
4. The WQI mathematical expression should be included in the text part
--- Response ---
Implemented. The revised Methods section includes the WQI mathematical expression with full symbol definitions and methodological references.
5. The ML model mathematical expressions are also not mentioned in the methodology section.
--- Response ---
Implemented. We added concise mathematical formulations for SVR, ANN, and the Random Forest regression framework in the methodology section.
6. Why only SVR and RFR model based weight have been mentioned in Table 2. Why authors did not mention the weight from ANN model
--- Response ---
We clarified that ANN does not provide directly comparable global coefficient-style importance in the same way. To improve interpretability, we now report ANN feature influence using a model-agnostic method and explicitly distinguish interpretation scope across model types.
7. Authors should include the hyperparameter tuning optimal values during training process for each ML models in a form of table.
--- Response ---
Implemented. We added a table showing the search space and optimal hyperparameter values for ANN, SVR, and RFR.
8. The font size of the axes labelling should be increased for better visualization. (exam: figure 3,...)
--- Response ---
Implemented. Axis labels, tick labels, and legends were resized across figures to improve readability, including Figure 3.
9. The statistical details of considered input parameter should be included in a table form by mention mean, max, min, SD,...
--- Response ---
Implemented. We added a descriptive-statistics table for all input variables (mean, minimum, maximum, standard deviation, and sample size where applicable).
10. The RFR model exhibited superior prediction performance (r2=0.715). Here, the authors could try using a hybrid ML model to obtain higher prediction accuracy.
--- Response ---
Thank you for this suggestion. In this revision, we prioritized transparency and reproducibility of the core benchmark models. We have added hybrid-model exploration as a future-work direction under the same temporal validation protocol.
