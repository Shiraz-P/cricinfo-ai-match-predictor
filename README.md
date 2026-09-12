\# Cricinfo AI Cricket Match Prediction Project



\## 1. Project Overview



This project builds an end-to-end Machine Learning solution for predicting

the winner of a cricket match.



The project demonstrates the complete ML lifecycle:



Data Acquisition

→ Data Cleaning

→ Feature Engineering

→ Model Training

→ Model Testing

→ Probability Prediction

→ Model Evaluation

→ Calibration Analysis

→ Drift Analysis

→ Error Analysis

→ Feature Reliability Analysis

→ Champion vs Challenger

→ Shadow Testing

→ Model Monitoring





\## 2. Project Objective



The objective is to predict which of two cricket teams is more likely

to win a match using historical cricket data.



The project also demonstrates how a Machine Learning model can be

evaluated and monitored before replacing an existing production model.





\## 3. Technology Used



Python



Pandas



NumPy



Scikit-Learn



SciPy



Joblib



CSV datasets





\## 4. Machine Learning Type



This project uses:



Supervised Machine Learning



The problem is:



Binary Classification



The primary algorithm is:



Logistic Regression





\## 5. Target Variable



The target variable is:



team1\_won



Meaning:



1 = Team 1 won



0 = Team 2 won





\## 6. Baseline Model Features



The production Champion model v1.0 uses 9 features:



1\. toss\_winner\_is\_team1



2\. team1\_historical\_win\_rate



3\. team2\_historical\_win\_rate



4\. team1\_recent\_win\_rate



5\. team2\_recent\_win\_rate



6\. team1\_h2h\_win\_rate



7\. team2\_h2h\_win\_rate



8\. team1\_venue\_win\_rate



9\. team2\_venue\_win\_rate





\## 7. Historical Win Rate



Historical win rate measures how often a team has won its previous

matches.



Example:



If Pakistan played 100 previous matches and won 60:



Historical Win Rate = 60%





\## 8. Recent Form



Recent win rate measures performance in the team's most recent matches.



The current project uses the last 5 matches.



Example:



4 wins from last 5 matches:



Recent Win Rate = 80%





\## 9. Head-to-Head Performance



Head-to-head win rate measures how a team historically performed

against the specific opponent.



Example:



Pakistan vs India



If Pakistan won 3 out of 10 previous matches:



Pakistan H2H Win Rate = 30%





\## 10. Venue Performance



Venue win rate measures how successfully a team has historically

performed at the selected venue.





\## 11. Toss Feature



The toss feature is represented as:



toss\_winner\_is\_team1



1 = Team 1 won the toss



0 = Team 2 won the toss





\## 12. Feature Engineering



Raw match data is converted into numerical information that can be

understood by the Machine Learning algorithm.



This process is called Feature Engineering.





\## 13. Chronological Train/Test Split



Because cricket matches occur over time, the project uses a

chronological split rather than randomly mixing old and new matches.



Earlier matches are used for training.



Later matches are used for testing.



This better represents a real prediction scenario.





\## 14. Baseline Model



Model:



Logistic Regression



Model Version:



1.0



Feature Count:



9



Holdout Test Accuracy:



71.82%





\## 15. Model Persistence



The trained model is stored using Joblib.



Production model:



model\\cricket\_model.pkl



This allows prediction scripts to load the trained model without

training it every time.





\## 16. Prediction System



The prediction script accepts:



Team 1



Team 2



Venue or City



Toss Winner





It automatically calculates historical features and sends them to the

trained model.





\## 17. Input Normalization



The prediction system supports user-friendly inputs.



Example:



newzealand



can resolve to:



New Zealand





Likewise:



lords



can resolve to:



Lord's





City inputs can also resolve to known venues.





\## 18. Fuzzy Matching



Python difflib is used to detect possible spelling mistakes.



Example:



pakstan



may suggest:



Pakistan





This improves usability without changing the ML model.





\## 19. Prediction Probability



Logistic Regression produces probabilities in addition to the predicted

winner.



Example:



India: 58%



Pakistan: 42%





The higher probability determines the predicted winner.





\## 20. Prediction Confidence



The project converts probabilities into simple confidence levels.



HIGH:



70% or greater



MEDIUM:



60% to less than 70%



LOW:



Less than 60%





Confidence is not the same thing as accuracy.



It represents how strongly the model favors one prediction.





\## 21. Prediction Logging



Predictions are stored in:



data\\prediction\_log.csv





This enables future comparison between:



Predicted Winner



and



Actual Winner





\## 22. Ground Truth



Ground truth means the actual observed outcome.



For this project:



Ground Truth = Actual Match Winner





Without ground truth, live model accuracy cannot be calculated.





\## 23. Model Monitoring



monitor\_model.py monitors completed predictions.



It reports:



Total Predictions



Pending Predictions



Completed Predictions



Correct Predictions



Incorrect Predictions



Live Accuracy



Accuracy by Confidence



Accuracy by Model Version





\## 24. Retraining Readiness



retraining\_check.py determines whether enough live results exist to

consider retraining.



Current monitoring rule:



Minimum completed predictions = 30





This is a project operational rule, not a universal ML rule.





\## 25. Feature Drift



Feature drift occurs when the statistical characteristics of model

inputs change over time.



drift\_analysis.py compares historical and recent feature behavior.





\## 26. Mean-Based Drift Analysis



The first drift test compares average feature values between historical

and recent periods.



Current project rules:



Less than 5 percentage points:



LOW / STABLE





5 to less than 10:



MODERATE / WATCH





10 or more:



HIGH / POSSIBLE DRIFT





These thresholds are project monitoring rules.





\## 27. Statistical Drift



statistical\_drift.py uses the Kolmogorov-Smirnov test.



The KS test compares historical and recent feature distributions.





\## 28. P-Value



The p-value helps determine whether the observed distribution

difference is statistically significant.



Project significance threshold:



0.05





A low p-value indicates evidence that the distributions differ.





\## 29. KS Statistic



The KS statistic measures the size of the difference between two

distributions.



A low p-value alone does not mean the model must be retrained.



Practical effect size must also be considered.





\## 30. Probability Calibration



Calibration measures whether predicted probabilities correspond to

observed outcomes.



Example:



If matches predicted around 70% are actually won around 70% of the

time, the probabilities are well calibrated.





\## 31. Brier Score



Brier Score evaluates probability prediction quality.



Lower is better.



0 represents perfect probability predictions.





The baseline model achieved approximately:



Brier Score = 0.199





\## 32. Calibration Methods



Two calibration approaches were tested:



Sigmoid Calibration



and



Isotonic Calibration





\## 33. Sigmoid Calibration



Sigmoid calibration is also commonly associated with Platt-style

probability scaling.



It applies a smooth parametric correction to model probabilities.





\## 34. Isotonic Calibration



Isotonic calibration is a more flexible non-parametric calibration

method.



It can model more complex probability corrections but can overfit when

calibration data is limited.





\## 35. Calibration Validation



Multiple time periods were tested rather than selecting a calibration

method from one test period.



Average validation results showed:



Sigmoid calibration produced the best average Brier Score among the

tested calibration approaches.





\## 36. Temporal Validation



Temporal validation evaluates the model across multiple chronological

periods.



This helps determine whether improvements generalize across time.





\## 37. Temporal Error Analysis



temporal\_error\_analysis.py identifies where the model performs well or

poorly.



Analysis includes:



Accuracy by Year



Accuracy by Confidence



Similar-Strength Matches



Toss Impact



Hardest Teams to Predict



Most Confident Wrong Predictions





\## 38. Feature Reliability



A percentage based on many historical matches is generally more

trustworthy than a percentage based on only one or two matches.



Therefore, sample size was analyzed for:



Historical records



Head-to-head records



Venue records





\## 39. Sample Size Reliability



Feature reliability analysis showed that many H2H and venue rates are

based on small samples.



For example:



1 match and 1 win = 100%



but this is much less reliable than:



80 wins from 100 matches = 80%





\## 40. Reliability Features



Additional reliability features were created using sample sizes.



The experiment tested:



Historical reliability



H2H reliability



Venue reliability





\## 41. Log Transformation



Sample sizes were transformed using logarithmic scaling.



This reduces the influence of extremely large sample counts while

preserving useful information about reliability.





\## 42. Ablation Testing



Ablation testing means adding or removing feature groups to determine

which components actually improve the model.



The project tested:



Baseline 9 Features



Baseline + Historical Reliability



Baseline + H2H Reliability



Baseline + Venue Reliability



Baseline + All Reliability





\## 43. Best Reliability Candidate



The strongest candidate was:



Baseline + Historical Reliability





Feature Count:



11





Holdout Accuracy:



72.41%





Brier Score:



0.1928





This improved both classification accuracy and probability quality on

the holdout test.





\## 44. Rolling Validation



The reliability candidate was tested across multiple chronological

folds.



Results:



Baseline Average Accuracy:



64.70%





Candidate Average Accuracy:



66.28%





Baseline Average Brier:



0.2199





Candidate Average Brier:



0.2128





The candidate won:



Accuracy: 5 out of 5 folds



Brier Score: 5 out of 5 folds





\## 45. Champion Model



Current production model:



Version 1.0



Features:



9



Algorithm:



Logistic Regression





This model remains unchanged while the challenger is evaluated.





\## 46. Challenger Model



Candidate model:



Version 1.1-candidate



Status:



CHALLENGER



Features:



11





The additional features represent historical sample-size reliability.





Model file:



model\\cricket\_model\_v1\_1\_candidate.pkl





\## 47. Champion-Challenger Strategy



Champion-Challenger testing compares:



Current trusted model



against



New candidate model





The challenger is not immediately allowed to replace the champion.





\## 48. Shadow Testing



Shadow testing allows both models to process the same match input.



Their predictions are compared while the production champion remains

unchanged.





\## 49. Shadow Comparison Log



Shadow comparisons are stored in:



data\\shadow\_comparison\_log.csv





The log includes:



Champion Prediction



Challenger Prediction



Probabilities



Confidence



Model Agreement



Probability Difference



Actual Winner



Result Status





\## 50. Pending Prediction



PENDING means the prediction has been made but the real match outcome

has not yet been recorded.





\## 51. Completed Prediction



COMPLETED means the actual winner is known and the prediction can be

evaluated against ground truth.





\## 52. Pandas dtype Inference



Pandas automatically determines column data types when reading CSV

files.



A completely blank column may be interpreted as float64 because missing

values are represented as NaN.





\## 53. NaN



NaN means Not a Number.



Pandas commonly uses NaN to represent missing values.





\## 54. Explicit Data-Type Handling



Operational log columns such as actual\_winner must support text.



Therefore the project explicitly normalizes result columns before

updating them.



This prevents errors such as trying to insert the string "Pakistan"

into a float64 column.





\## 55. Shadow Monitoring Dashboard



shadow\_monitor.py provides an operational comparison between Champion

and Challenger.



It reports:



Total Shadow Predictions



Completed Predictions



Pending Predictions



Agreement Rate



Disagreement Rate



Probability Difference



Champion Accuracy



Challenger Accuracy



Confidence Distribution



Disagreement Cases



Promotion Readiness





\## 56. Current Shadow Monitoring State



Current shadow predictions:



2



Completed:



0



Pending:



2



Agreement Rate:



100%





Average Probability Difference:



2.14 percentage points





Maximum Difference:



3.03 percentage points





Minimum Difference:



1.26 percentage points





These numbers are based on only two predictions and are not sufficient

for a promotion decision.





\## 57. Promotion Readiness



Current project rule:



Minimum completed shadow predictions:



30





Current completed predictions:



0





Therefore:



NOT READY FOR PROMOTION





\## 58. Historical Evidence vs Live Evidence



Historical validation and live shadow validation are separate forms of

evidence.



Historical validation currently favors Challenger v1.1.



Live shadow validation does not yet have enough completed outcomes.



Therefore the production model remains Champion v1.0.





\## 59. Model Promotion



A challenger should only be considered for promotion when sufficient

evidence shows that it performs reliably.



Promotion should consider:



Historical Validation



Live Accuracy



Probability Quality



Calibration



Drift



Error Analysis



Operational Stability



Business Impact





\## 60. Current Model Status



Champion:



v1.0



9 Features



Production





Challenger:



v1.1-candidate



11 Features



Shadow / Challenger





Production model has NOT been replaced.





\## 61. Project Workflow



The current project workflow is:



Raw Cricket Data

&#x20;       |

&#x20;       v

Data Cleaning

&#x20;       |

&#x20;       v

Feature Engineering

&#x20;       |

&#x20;       v

matches\_features.csv

&#x20;       |

&#x20;       v

Chronological Train/Test Split

&#x20;       |

&#x20;       v

Logistic Regression

&#x20;       |

&#x20;       v

Baseline Model v1.0

&#x20;       |

&#x20;       +----------------------+

&#x20;       |                      |

&#x20;       v                      v

Model Evaluation        Error / Drift Analysis

&#x20;       |                      |

&#x20;       v                      v

Calibration          Feature Reliability

&#x20;       |                      |

&#x20;       +-----------+----------+

&#x20;                   |

&#x20;                   v

&#x20;            Ablation Testing

&#x20;                   |

&#x20;                   v

&#x20;        Reliability Candidate

&#x20;                   |

&#x20;                   v

&#x20;           Challenger v1.1

&#x20;                   |

&#x20;                   v

&#x20;       Champion vs Challenger

&#x20;                   |

&#x20;                   v

&#x20;            Shadow Testing

&#x20;                   |

&#x20;                   v

&#x20;       shadow\_comparison\_log.csv

&#x20;                   |

&#x20;                   v

&#x20;             Ground Truth

&#x20;                   |

&#x20;                   v

&#x20;         Shadow Model Monitor

&#x20;                   |

&#x20;                   v

&#x20;          Promotion Decision





\## 62. Important Project Scripts



predict\_match.py



Purpose:

Run normal match predictions using the production model.





update\_result.py



Purpose:

Add actual match results to prediction history.





monitor\_model.py



Purpose:

Monitor production prediction performance.





retraining\_check.py



Purpose:

Determine whether enough completed live predictions exist for a

retraining decision.





drift\_analysis.py



Purpose:

Perform simple mean-based feature drift analysis.





statistical\_drift.py



Purpose:

Perform KS statistical distribution testing.





calibration\_analysis.py



Purpose:

Evaluate probability calibration and Brier Score.





compare\_calibration.py



Purpose:

Compare original, sigmoid-calibrated, and isotonic-calibrated models.





validate\_calibration.py



Purpose:

Validate calibration methods across multiple chronological periods.





temporal\_error\_analysis.py



Purpose:

Analyze where and when model prediction errors occur.





feature\_reliability.py



Purpose:

Measure sample sizes supporting historical, H2H, and venue features.





reliability\_model\_test.py



Purpose:

Test a model containing additional reliability features.





reliability\_ablation.py



Purpose:

Determine which reliability feature groups actually add value.





validate\_reliability\_candidate.py



Purpose:

Validate the best reliability candidate across multiple time periods.





train\_candidate\_v11.py



Purpose:

Train and package model v1.1-candidate.





compare\_models\_live.py



Purpose:

Run Champion v1.0 and Challenger v1.1 on the same match and store a

shadow comparison.





update\_shadow\_result.py



Purpose:

Add actual match outcomes to shadow predictions.





shadow\_monitor.py



Purpose:

Monitor Champion vs Challenger shadow performance and promotion

readiness.





\## 63. Important Data Files



data\\matches\_features.csv



Feature-engineered ML dataset.





data\\prediction\_log.csv



Production prediction history.





data\\shadow\_comparison\_log.csv



Champion vs Challenger shadow comparison history.





data\\temporal\_error\_analysis.csv



Saved temporal error analysis.





data\\feature\_reliability\_analysis.csv



Feature reliability and sample-size analysis.





\## 64. Model Files



model\\cricket\_model.pkl



Champion v1.0 production model.





model\\cricket\_model\_v1\_1\_candidate.pkl



Challenger v1.1 candidate model.





\## 65. Current Project Achievement



The project has progressed beyond simply training a classification

model.



It now demonstrates:



Machine Learning



Feature Engineering



Model Evaluation



Probability Analysis



Calibration



Drift Detection



Error Analysis



Feature Reliability



Model Versioning



Champion-Challenger Testing



Shadow Deployment



Ground-Truth Collection



Model Monitoring



Promotion Governance





\## 66. Academic Assignment Mapping



The assignment requirement includes:



Data acquisition / scraping or open dataset



COMPLETED





Pandas / NumPy feature engineering



COMPLETED





Scikit-Learn classification or regression



COMPLETED





Classification selected:



Logistic Regression





Model evaluation:



COMPLETED





Advanced model analysis:



COMPLETED





Operational monitoring demonstration:



COMPLETED





\## 67. Current Conclusion



The production Champion v1.0 achieves:



71.82% holdout accuracy.





The Challenger v1.1 achieved:



72.41% holdout accuracy



and



0.1928 Brier Score.





Rolling validation also favored the Challenger across all five tested

time periods.



However, the challenger has not replaced the production model because

live shadow ground-truth evidence is still insufficient.



This demonstrates an important Machine Learning engineering principle:



A better experimental result does not automatically justify production

deployment.



Models should be validated, monitored, compared, and promoted using

evidence.

