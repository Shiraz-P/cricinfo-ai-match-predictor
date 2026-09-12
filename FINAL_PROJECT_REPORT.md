\# Cricinfo AI Cricket Match Prediction System

\## Final Project Report



\## 1. Executive Summary



This project develops an end-to-end Machine Learning system for predicting

cricket match winners using historical match information.



The project began with dataset construction and exploratory analysis and

progressed through feature engineering, Logistic Regression model training,

chronological validation, probability analysis, calibration, feature drift,

error analysis, feature reliability, candidate-model development,

Champion-Challenger comparison, shadow testing, and model monitoring.



The current production Champion is Logistic Regression model v1.0 using

9 features.



Its holdout test accuracy is:



71.82%



A reliability-aware Challenger v1.1 was subsequently developed using

11 features.



Its holdout performance is:



Accuracy: 72.41%



Brier Score: 0.1928



Rolling chronological validation also showed that the Challenger

outperformed the baseline across all five tested folds for both accuracy

and Brier Score.



The Challenger has deliberately not replaced the Champion.



It is currently being evaluated using a shadow-testing approach so that

additional live ground-truth evidence can be collected before a model

promotion decision is made.





\## 2. Problem Statement



Cricket match outcomes depend on multiple historical and contextual factors.



Examples include:



\- Overall team performance

\- Recent team form

\- Previous head-to-head performance

\- Venue performance

\- Toss outcome



The objective of this project is to determine whether these historical

signals can be transformed into numerical features and used by a

Machine Learning model to estimate the probability of each team winning.





\## 3. Project Objective



The primary objective is to build a Machine Learning classification

system that predicts the winner between two cricket teams.



The project also aims to demonstrate the complete Machine Learning lifecycle:



Data Acquisition

\-> Data Preparation

\-> Feature Engineering

\-> Model Training

\-> Model Validation

\-> Prediction

\-> Probability Evaluation

\-> Error Analysis

\-> Calibration Analysis

\-> Drift Detection

\-> Feature Reliability

\-> Model Improvement

\-> Champion-Challenger Testing

\-> Shadow Testing

\-> Model Monitoring





\## 4. Type of Machine Learning



This project uses:



Supervised Machine Learning



The task is:



Binary Classification



The target variable is:



team1\_won



where:



1 = Team 1 won



0 = Team 2 won





\## 5. Technology Stack



The project uses:



Python



Pandas



NumPy / numerical processing



Scikit-Learn



SciPy



Joblib



CSV-based datasets



Windows Command Prompt



Notepad for lightweight code/document editing





\## 6. Dataset Pipeline



Dataset construction and analysis are handled through scripts including:



build\_dataset.py



analyze\_dataset.py



feature\_engineering.py



The feature-engineered dataset is stored as:



data\\matches\_features.csv



The final dataset currently contains:



3,424 matches



The matches span approximately:



17-Feb-2005



to



09-Sep-2026





\## 7. Data Preparation



Pandas is used to load and manipulate the cricket dataset.



Important preparation steps include:



\- Loading CSV data

\- Converting dates to datetime

\- Sorting matches chronologically

\- Handling missing information

\- Creating numerical model features

\- Preparing the target variable



Chronological sorting is particularly important because the model should

learn from earlier matches before being evaluated on later matches.





\## 8. Feature Engineering



Feature Engineering converts raw cricket information into numerical values

that can be processed by a Machine Learning algorithm.



The Champion model uses the following 9 features:



1\. toss\_winner\_is\_team1



2\. team1\_historical\_win\_rate



3\. team2\_historical\_win\_rate



4\. team1\_recent\_win\_rate



5\. team2\_recent\_win\_rate



6\. team1\_h2h\_win\_rate



7\. team2\_h2h\_win\_rate



8\. team1\_venue\_win\_rate



9\. team2\_venue\_win\_rate





\## 9. Historical Win Rate



Historical Win Rate measures a team's overall success in previous matches.



For example:



Previous Matches = 100



Wins = 60



Historical Win Rate = 60%





\## 10. Recent Form



Recent Win Rate measures a team's recent performance.



The current implementation uses the last five matches.



For example:



Wins in last five matches = 4



Recent Win Rate = 80%





\## 11. Head-to-Head Performance



Head-to-Head, or H2H, measures previous results between the two teams.



For example:



Pakistan vs India



Pakistan wins = 3



India wins = 7



Total matches = 10



Pakistan H2H Win Rate = 30%



India H2H Win Rate = 70%





\## 12. Venue Performance



Venue Win Rate measures how successfully a team has historically performed

at the selected ground.



For example:



Australia



at



Sydney Cricket Ground



The model calculates Australia's historical win rate at that venue.





\## 13. Toss Information



The toss is represented by:



toss\_winner\_is\_team1



Values are:



1 = Team 1 won toss



0 = Team 2 won toss



This converts categorical cricket information into a numerical feature.





\## 14. Model Selection



The primary model selected for the project is:



Logistic Regression



Logistic Regression is suitable for binary classification.



Instead of producing only a winner, it can also estimate probabilities.



For example:



India = 63%



Pakistan = 37%





\## 15. Training and Testing Strategy



The project uses chronological splitting.



Earlier matches are used for training.



Later matches are used for testing.



This is preferable to randomly mixing matches because the real objective is:



Learn from the past



and



Predict future outcomes.





\## 16. Champion Model v1.0



The current production model is:



Model Type:



Logistic Regression



Model Version:



1.0



Feature Count:



9



Holdout Test Accuracy:



71.82%



The model is stored as:



model\\cricket\_model.pkl





\## 17. Model Persistence



Joblib is used to save and load the trained model.



This means the prediction application does not need to retrain the model

every time it starts.



The saved package also stores metadata including:



\- Model version

\- Model type

\- Feature list

\- Feature count

\- Test accuracy





\## 18. Prediction Application



predict\_match.py provides the user-facing prediction workflow.



The user enters:



Team 1



Team 2



Venue or City



Toss Winner



The application then automatically calculates the required features and

sends them to the trained model.





\## 19. User Input Validation



Input validation was added to improve usability.



Examples include:



pakstan



can suggest:



Pakistan



and:



newzealand



can resolve to:



New Zealand





Venue normalization was also implemented.



For example:



lords



resolves to:



Lord's





\## 20. Fuzzy Matching



Python's difflib functionality is used to detect similar team names when

a spelling mistake occurs.



This improves the application interface but is separate from the Machine

Learning algorithm itself.





\## 21. Prediction Probability



The model uses:



predict\_proba()



to calculate probability estimates.



Example:



Pakistan Win Probability = 41%



India Win Probability = 59%



The higher probability determines the predicted winner.





\## 22. Confidence Levels



The project translates model probabilities into operational confidence

levels.



HIGH:



70% or higher



MEDIUM:



60% to less than 70%



LOW:



Less than 60%



Confidence represents how strongly the model favors its selected outcome.



It does not guarantee that the prediction is correct.





\## 23. Prediction Logging



Predictions are logged in:



data\\prediction\_log.csv



This allows predictions to later be compared with actual results.





\## 24. Ground Truth



Ground Truth means the real observed outcome.



For this project:



Ground Truth = Actual Match Winner



Ground truth is required before a live prediction can be classified as

correct or incorrect.





\## 25. Model Monitoring



monitor\_model.py monitors prediction performance.



It reports information such as:



\- Total predictions

\- Pending predictions

\- Completed predictions

\- Correct predictions

\- Incorrect predictions

\- Live accuracy

\- Accuracy by confidence

\- Accuracy by model version





\## 26. Retraining Readiness



retraining\_check.py checks whether enough completed predictions exist to

consider a retraining decision.



The current project rule requires:



30 completed predictions



before making a retraining decision.



This is a project monitoring rule rather than a universal Machine Learning

standard.





\## 27. Error Analysis



error\_analysis.py and temporal\_error\_analysis.py investigate where the

model performs poorly.



This is important because overall accuracy alone does not explain the

model's weaknesses.





\## 28. Temporal Error Analysis



The chronological test set contained:



685 matches



Overall Accuracy:



71.82%



Accuracy during 2025:



75.29%



Accuracy during 2026:



69.67%



This indicates that predictive performance varies over time.





\## 29. Accuracy by Confidence



The temporal analysis produced:



HIGH Confidence:



160 matches



Accuracy = 85.62%





MEDIUM Confidence:



237 matches



Accuracy = 75.95%





LOW Confidence:



288 matches



Accuracy = 60.76%



This suggests that the confidence signal provides useful information.





\## 30. Similar-Strength Matches



The model was less accurate when the competing teams had similar

historical strength.



Similar-strength matches:



259



Accuracy:



62.93%



This identifies an important difficult prediction segment.





\## 31. Hard-to-Predict Teams



Temporal error analysis identified teams with relatively low prediction

accuracy.



Examples included:



Qatar



Czech Republic



Eswatini



Nepal



Zambia



Sweden



Zimbabwe



Hong Kong



Namibia



New Zealand



This indicates that model performance is not uniform across all teams.





\## 32. Feature Drift



Feature Drift occurs when the statistical characteristics of model input

features change over time.



The project compares historical and recent data to identify such changes.





\## 33. Mean-Based Drift Analysis



drift\_analysis.py compares historical and recent feature averages.



Historical matches:



2,739



Recent matches:



685



The initial mean-based analysis found all monitored features within the

project's LOW / STABLE range.



The largest mean difference was approximately:



2.63 percentage points



for:



team1\_venue\_win\_rate





\## 34. Statistical Drift Analysis



Because averages alone cannot describe complete distributions,

statistical\_drift.py uses the:



Kolmogorov-Smirnov Test



also called:



KS Test





\## 35. KS Test



The KS Test compares two distributions.



The project considers:



KS Statistic



and



P-Value



The p-value determines whether there is statistical evidence of a

distributional change.



The KS statistic indicates the magnitude of that difference.





\## 36. Statistical Drift Results



The statistical test identified:



7 statistically significant feature changes



and:



1 feature without statistically significant change



Using the project's practical KS severity rules:



LOW = 1 feature



MODERATE = 6 features



HIGH = 1 feature



Overall status:



INVESTIGATE





\## 37. Statistical vs Practical Significance



An important learning from the drift analysis is that:



Statistical significance



does not necessarily mean:



Operationally important change.



Large datasets can produce low p-values even for relatively modest

distribution changes.



Therefore both statistical evidence and practical effect size should be

considered.





\## 38. Probability Calibration



Probability Calibration evaluates whether model probabilities correspond

to actual outcome frequencies.



For example:



If a model repeatedly predicts approximately 70% probability, a

well-calibrated model should observe that outcome roughly 70% of the time.





\## 39. Brier Score



Brier Score evaluates probability prediction quality.



Lower is better.



A score of:



0



represents perfect probability predictions.



The baseline model's evaluated Brier Score was approximately:



0.199





\## 40. Calibration Analysis



The project compared:



Original Logistic Regression



Sigmoid Calibration



Isotonic Calibration



In one final holdout comparison:



Original Logistic Regression:



Accuracy = 72.26%



Brier = 0.2032





Sigmoid Calibrated:



Accuracy = 71.82%



Brier = 0.1939





Isotonic Calibrated:



Accuracy = 71.68%



Brier = 0.1927



Isotonic therefore achieved the best Brier Score on that individual

comparison.





\## 41. Sigmoid Calibration



Sigmoid calibration is a smooth parametric probability-correction method.



It is associated with Platt-style probability scaling.



It generally applies a smooth transformation to raw model probabilities.





\## 42. Isotonic Calibration



Isotonic calibration is a more flexible non-parametric method.



It can represent more complicated calibration relationships.



However, it can be more vulnerable to overfitting when calibration data

is limited.





\## 43. Time-Based Calibration Validation



Calibration methods were then evaluated across multiple chronological

periods.



Average Original Accuracy:



64.91%



Average Original Brier:



0.2215





Average Sigmoid Accuracy:



64.35%



Average Sigmoid Brier:



0.2195





Average Isotonic Accuracy:



64.07%



Average Isotonic Brier:



0.2250





Based on average Brier Score:



Sigmoid Calibrated



performed best across the tested periods.





\## 44. Feature Reliability



The project identified an important weakness in some engineered features.



A win rate does not communicate how many historical matches support the

percentage.



For example:



1 win from 1 match = 100%



and:



80 wins from 100 matches = 80%



The first value looks stronger but is based on much less evidence.





\## 45. Sample Size Analysis



feature\_reliability.py analyzed the number of observations supporting:



\- Historical win rates

\- H2H win rates

\- Venue win rates



The analysis showed that many H2H and venue statistics were based on

small sample sizes.





\## 46. H2H Reliability



For Team 1 H2H reliability:



LOW = 2,189 matches



MEDIUM = 1,002 matches



HIGH = 233 matches



This showed that H2H statistics frequently had limited supporting history.





\## 47. Venue Reliability



Team 1 Venue Reliability:



LOW = 2,544



MEDIUM = 764



HIGH = 116





Team 2 Venue Reliability:



LOW = 2,667



MEDIUM = 656



HIGH = 101



Venue statistics therefore frequently contain limited historical evidence.





\## 48. Reliability-Aware Model



An experimental reliability-aware model was developed using 14 features.



Baseline 9-feature model:



Accuracy = 71.82%



Brier Score = 0.1990





Reliability-aware 14-feature model:



Accuracy = 70.66%



Brier Score = 0.1953



The experimental model improved probability quality but reduced

classification accuracy.





\## 49. Ablation Testing



Rather than accepting or rejecting all reliability features together,

Ablation Testing was performed.



Ablation testing determines which feature groups actually contribute

useful information.



The following were compared:



Baseline 9



Baseline + Historical Reliability



Baseline + H2H Reliability



Baseline + Venue Reliability



Baseline + All Reliability





\## 50. Ablation Results



The best model was:



Baseline + Historical Reliability



Feature Count:



11



Accuracy:



72.41%



Brier Score:



0.1928



Accuracy improvement:



+0.58 percentage points



Brier improvement:



0.0063



This candidate improved both metrics on the holdout test.





\## 51. Rolling Candidate Validation



The candidate was then tested across five chronological validation folds.



Baseline Average Accuracy:



64.70%



Candidate Average Accuracy:



66.28%



Baseline Average Brier:



0.2199



Candidate Average Brier:



0.2128



Candidate accuracy wins:



5 out of 5



Candidate Brier wins:



5 out of 5





\## 52. Challenger Model v1.1



Following successful validation, a candidate model package was created.



Model Version:



1.1-candidate



Status:



CHALLENGER



Feature Count:



11



Holdout Accuracy:



72.41%



Brier Score:



0.1928



Model location:



model\\cricket\_model\_v1\_1\_candidate.pkl





\## 53. Champion-Challenger Strategy



The existing production model is called the:



Champion



The new candidate is called the:



Challenger



Current Champion:



v1.0



9 features





Current Challenger:



v1.1-candidate



11 features





The Challenger is not automatically promoted simply because historical

evaluation is better.





\## 54. Shadow Testing



Shadow Testing allows both models to process the same input while keeping

the current Champion unchanged.



The system compares:



\- Predicted winner

\- Probabilities

\- Confidence

\- Model agreement

\- Probability difference





\## 55. Shadow Comparison Logging



Shadow comparisons are stored in:



data\\shadow\_comparison\_log.csv



The log later allows both models to be compared with actual match outcomes.





\## 56. Current Shadow Results



Current shadow predictions:



2



Completed:



0



Pending:



2



Champion-Challenger Agreement:



100%



Average Probability Difference:



2.14 percentage points



Maximum Probability Difference:



3.03 percentage points



Minimum Probability Difference:



1.26 percentage points





\## 57. Interpretation of Current Shadow Results



The current 100% agreement rate must not be interpreted as proof that the

models are equivalent.



Only two shadow predictions currently exist.



Additionally, neither prediction has a completed ground-truth result.



Therefore live model accuracy cannot yet be calculated.





\## 58. Promotion Readiness



shadow\_monitor.py applies a current project requirement of:



30 completed shadow predictions



before considering live model promotion.



Current completed predictions:



0



Current promotion status:



NOT READY FOR PROMOTION





\## 59. Historical vs Live Evidence



Historical validation and live shadow validation answer different

questions.



Historical Validation asks:



How did the model perform on historical unseen data?



Live Shadow Validation asks:



How does the candidate perform on newly collected predictions?



Historical evidence currently favors v1.1.



Live evidence is still insufficient.





\## 60. Model Governance



The project deliberately protects the production model.



Production Champion:



model\\cricket\_model.pkl



Experimental Challenger:



model\\cricket\_model\_v1\_1\_candidate.pkl



This separation prevents experiments from accidentally replacing the

trusted production model.





\## 61. MLOps Concepts Demonstrated



Although this is an educational project, it now demonstrates several

introductory MLOps concepts:



\- Model packaging

\- Model versioning

\- Feature schema validation

\- Model monitoring

\- Drift detection

\- Probability monitoring

\- Ground-truth collection

\- Champion-Challenger strategy

\- Shadow testing

\- Promotion gates

\- Retraining readiness





\## 62. Current Project Architecture



The completed architecture is:



Cricket Data

&#x20;    |

&#x20;    v

Dataset Construction

&#x20;    |

&#x20;    v

Data Analysis

&#x20;    |

&#x20;    v

Feature Engineering

&#x20;    |

&#x20;    v

matches\_features.csv

&#x20;    |

&#x20;    v

Chronological Training

&#x20;    |

&#x20;    v

Logistic Regression

&#x20;    |

&#x20;    v

Champion v1.0

&#x20;    |

&#x20;    +----------------------------+

&#x20;    |                            |

&#x20;    v                            v

Evaluation                  Advanced Analysis

&#x20;                            |

&#x20;                            +-- Calibration

&#x20;                            |

&#x20;                            +-- Drift

&#x20;                            |

&#x20;                            +-- Errors

&#x20;                            |

&#x20;                            +-- Reliability

&#x20;                                  |

&#x20;                                  v

&#x20;                             Ablation Test

&#x20;                                  |

&#x20;                                  v

&#x20;                          Challenger v1.1

&#x20;                                  |

&#x20;    +-----------------------------+

&#x20;    |

&#x20;    v

Champion vs Challenger

&#x20;    |

&#x20;    v

Shadow Testing

&#x20;    |

&#x20;    v

Shadow Comparison Log

&#x20;    |

&#x20;    v

Actual Match Results

&#x20;    |

&#x20;    v

Ground Truth

&#x20;    |

&#x20;    v

Shadow Monitoring

&#x20;    |

&#x20;    v

Promotion Readiness

&#x20;    |

&#x20;    v

Future Promotion Decision





\## 63. Key Project Scripts



build\_dataset.py



Builds the cricket dataset.





analyze\_dataset.py



Performs dataset-level analysis.





feature\_engineering.py



Creates Machine Learning features.





train\_model.py



Trains and packages the baseline model.





validate\_model.py



Validates model performance.





error\_analysis.py



Analyzes prediction errors.





predict\_match.py



Runs user-facing match predictions.





update\_result.py



Adds actual results to prediction records.





monitor\_model.py



Monitors live prediction results.





retraining\_check.py



Checks retraining readiness.





drift\_analysis.py



Performs mean-based feature drift analysis.





statistical\_drift.py



Performs KS statistical drift analysis.





calibration\_analysis.py



Evaluates probability calibration.





compare\_calibration.py



Compares calibration methods.





validate\_calibration.py



Performs chronological calibration validation.





temporal\_error\_analysis.py



Analyzes prediction errors over time.





feature\_reliability.py



Measures the reliability/sample size of engineered statistics.





reliability\_model\_test.py



Tests reliability-aware features.





reliability\_ablation.py



Tests reliability feature groups separately.





validate\_reliability\_candidate.py



Performs rolling validation of the best reliability candidate.





train\_candidate\_v11.py



Packages Challenger v1.1.





compare\_models\_live.py



Runs Champion and Challenger simultaneously.





update\_shadow\_result.py



Adds actual outcomes to shadow comparisons.





shadow\_monitor.py



Monitors Champion-Challenger shadow performance.





\## 64. Project Strengths



Major strengths of the project include:



1\. Chronological rather than random validation.



2\. Multiple model evaluation metrics.



3\. Probability quality analysis rather than accuracy alone.



4\. Feature drift monitoring.



5\. Statistical distribution testing.



6\. Temporal error analysis.



7\. Investigation of feature reliability.



8\. Ablation testing before adding features.



9\. Rolling validation before candidate creation.



10\. Separation between production and experimental models.



11\. Champion-Challenger architecture.



12\. Shadow monitoring and promotion controls.





\## 65. Current Limitations



The project also has important limitations.



First, cricket outcomes depend on information not currently represented

by the 9 or 11 model features.



Potential missing information includes:



\- Player selection

\- Player form

\- Injuries

\- Batting order

\- Bowling strength

\- Weather

\- Pitch conditions

\- Home advantage

\- Match format differences

\- Tournament importance



Second, some historical H2H and venue statistics have small sample sizes.



Third, only two live shadow predictions currently exist.



Fourth, no completed live shadow ground-truth outcomes are available yet.



Therefore live promotion decisions cannot currently be justified.





\## 66. Future Improvements



Potential future improvements include:



1\. Add player-level features.



2\. Add team ranking information.



3\. Add weather information.



4\. Add pitch/ground characteristics.



5\. Separate models by cricket format if appropriate.



6\. Compare additional classification algorithms.



7\. Perform systematic hyperparameter tuning.



8\. Improve probability calibration if validated.



9\. Add automated result ingestion.



10\. Build a web-based prediction interface.



11\. Add visual monitoring dashboards.



12\. Automate model retraining and promotion with appropriate controls.





\## 67. Assignment Requirement Mapping



The project satisfies the major assignment requirements.



Data Acquisition / Dataset:



COMPLETED





Pandas Data Processing:



COMPLETED





Feature Engineering:



COMPLETED





Scikit-Learn:



COMPLETED





Classification Model:



COMPLETED





Algorithm:



Logistic Regression





Model Evaluation:



COMPLETED





Prediction Application:



COMPLETED





Advanced Analysis:



COMPLETED





Monitoring Demonstration:



COMPLETED





\## 68. Main Learning Outcomes



Through this project, the following concepts were learned and implemented:



Supervised Learning



Classification



Logistic Regression



Features



Target Variable



Feature Engineering



Training Data



Testing Data



Chronological Split



Model Fitting



Prediction



Probability



Accuracy



Confidence



Brier Score



Calibration



Sigmoid Calibration



Isotonic Calibration



Temporal Validation



Error Analysis



Feature Drift



KS Test



P-Value



Statistical Significance



Practical Significance



Feature Reliability



Sample Size



Log Transformation



Ablation Testing



Model Versioning



Champion-Challenger



Shadow Testing



Ground Truth



Model Monitoring



Retraining



Promotion Gate



MLOps



Model Governance





\## 69. Final Results



Champion v1.0:



Algorithm:



Logistic Regression



Features:



9



Holdout Accuracy:



71.82%





Challenger v1.1:



Algorithm:



Logistic Regression



Features:



11



Holdout Accuracy:



72.41%



Brier Score:



0.1928





Rolling Validation:



Candidate Accuracy Wins:



5 / 5





Candidate Brier Wins:



5 / 5





Current Shadow Predictions:



2





Completed Shadow Results:



0





Promotion Status:



NOT READY FOR PROMOTION





\## 70. Final Conclusion



This project successfully demonstrates an end-to-end Machine Learning

solution for cricket match winner prediction.



The work progressed beyond simply training a model.



It includes:



Data Engineering



Feature Engineering



Model Training



Validation



Probability Evaluation



Calibration



Drift Detection



Error Analysis



Feature Reliability



Model Improvement



Champion-Challenger Testing



Shadow Deployment



Ground-Truth Tracking



Model Monitoring





The current Champion Logistic Regression model achieves 71.82% holdout

accuracy.



A new reliability-aware Challenger improves holdout accuracy to 72.41%

and achieves a Brier Score of 0.1928.



The Challenger also performed better across five rolling chronological

validation periods.



However, the project intentionally does not promote the Challenger yet

because sufficient live shadow ground-truth results have not been

collected.



The final lesson from the project is:



A Machine Learning model should not be judged by a single accuracy number.



A reliable ML system requires:



Good Data

\-> Good Features

\-> Proper Validation

\-> Error Analysis

\-> Probability Evaluation

\-> Monitoring

\-> Ground Truth

\-> Controlled Model Promotion

