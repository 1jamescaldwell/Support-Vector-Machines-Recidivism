# Support Vector Machines
# James Caldwell
# Spring 2024

# Required R packages and Directories
dir_data= 'https://mdporter.github.io/SYS6018/data/' # data directory
library(knitr)      # for nicer printing of tables with kable
library(e1071)      # for SVM
library(tidymodels) # for modeling and evaluation functions
library(tidyverse)  # functions for data manipulation  

# COMPAS Recidivism Prediction
# A recidivism risk model called COMPAS was the topic of a [ProPublica article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing/) on ML bias. Because the data and notebooks used for article was released on [github](https://github.com/propublica/compas-analysis), we can also evaluate the prediction bias (i.e., calibration). 
# This code will read in the *violent crime* risk score and apply the filtering used in the [analysis](https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb).

# Load data
df = read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years-violent.csv")

# Clean data
risk = df %>% 
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>% 
  filter(is_recid != -1) %>%
  filter(c_charge_degree != "O") %>%
  filter(v_score_text != 'N/A') %>% 
  transmute(
    age, age_cat,
    charge = ifelse(c_charge_degree == "F", "Felony", "Misdemeanor"),
    race,
    sex,                 
    priors_count = priors_count...15,
    score = v_decile_score,              # the risk score {1,2,...,10}
    outcome = two_year_recid...53        # outcome {1 = two year recidivate}
  )

# Problem 1: COMPAS risk score

# Assess the predictive bias in the COMPAS risk scores by evaluating the probability of recidivism, e.g. estimate $\Pr(Y = 1 \mid \text{Score}=x)$. Use any reasonable techniques (including Bayesian) to estimate the probability of recidivism for each risk score. 

# Create a table (e.g., data frame) that provides the following information:
# - The COMPASS risk score.
# - The point estimate of the probability of recidivism for each risk score.
# - 95% confidence or credible intervals for the probability (e.g., Using normal theory, bootstrap, or Bayesian techniques).

# Indicate the choices you made in estimation (e.g., state the prior if you used Bayesian methods).

# Group data by COMPASS risk score and calculate point estimate of probability of recidivism
recidivism_prob <- risk %>%
  group_by(score) %>%
  summarise(
    recidivism_rate = mean(outcome == 1),
    n = n()
  ) 

bootstrap_ci <- function(data, score_col, outcome_col, n_bootstrap = 50, confidence_level = 0.95) {
  bootstrapped_props <- data %>%
    group_by({{ score_col }}) %>%
    summarise(
      prob_lower_ci = mean({{ outcome_col }} == 1) - 1.96 * sqrt(mean({{ outcome_col }} == 1) * (1 - mean({{ outcome_col }} == 1)) / n()),
      prob_upper_ci = mean({{ outcome_col }} == 1) + 1.96 * sqrt(mean({{ outcome_col }} == 1) * (1 - mean({{ outcome_col }} == 1)) / n())
    )
  return(bootstrapped_props)
}

# Calculate bootstrap confidence intervals for probability of recidivism
bootstrap_ci <- bootstrap_ci(risk, score, outcome)

# Combine point estimates and confidence intervals into a single dataframe
recidivism_table <- inner_join(recidivism_prob, bootstrap_ci, by = "score")

print(recidivism_table, n = Inf)

# Make a plot of the risk scores and corresponding estimated probability of recidivism. 

# Put the risk score on the x-axis and estimate probability of recidivism on y-axis.
# Add the 95% confidence or credible intervals calculated in part a.
# Comment on the patterns you see.

ggplot(recidivism_table, aes(x = score, y = recidivism_rate)) +
  geom_point() +  # Add points for the estimated probabilities
  geom_errorbar(aes(ymin = prob_lower_ci, ymax = prob_upper_ci), width = 0.2) +  # Add error bars for confidence intervals
  labs(x = "Risk Score", y = "Estimated Probability of Recidivism") +  # Axis labels
  ggtitle("Estimated Probability of Recidivism vs. Risk Score") +  # Title
  theme_minimal()  

# Repeat the analysis, but this time do so for every race. Produce a set of plots (one per race) and comment on the patterns.

# Create a list to store the plots
plots <- list()

# Get unique race categories
race_categories <- unique(risk$race)

# Loop through each race category
for (race_i in race_categories) {
  # Subset the data for the current race
  race_data <- filter(risk, risk$race == race_i)
  
  # Estimate probability of recidivism and confidence intervals for this race
  recidivism_prob <- race_data %>%
    group_by(score) %>%
    summarise(recidivism_rate = mean(outcome == 1),  # Probability of recidivism
              n = n(),
              lower_ci = binom.test(sum(outcome == 1), n, conf.level = 0.95)$conf.int[1],  # Lower bound of CI
              upper_ci = binom.test(sum(outcome == 1), n, conf.level = 0.95)$conf.int[2])  # Upper bound of CI
  
  # Same function as part a
  bootstrap_ci <- function(data, score_col, outcome_col, n_bootstrap = 50, confidence_level = 0.95) {
    bootstrapped_props <- data %>%
      group_by({{ score_col }}) %>%
      summarise(
        prob_lower_ci = mean({{ outcome_col }} == 1) - 1.96 * sqrt(mean({{ outcome_col }} == 1) * (1 - mean({{ outcome_col }} == 1)) / n()),
        prob_upper_ci = mean({{ outcome_col }} == 1) + 1.96 * sqrt(mean({{ outcome_col }} == 1) * (1 - mean({{ outcome_col }} == 1)) / n())
      )
    return(bootstrapped_props)
  }
  
  bootstrap_ci_i <- bootstrap_ci(race_data, score, outcome)
  
  recidivism_table <- inner_join(recidivism_prob, bootstrap_ci_i, by = "score")
  
  print(race_i)
  # Print the recidivism table for the current race
  print(recidivism_table, n = Inf)
  
  # Create the plot for the current race
  p <- ggplot(recidivism_table, aes(x = score, y = recidivism_rate)) +
    geom_point() +  # Add points for the estimated probabilities
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.2) +  # Add error bars for confidence intervals
    labs(x = "Risk Score", y = "Estimated Probability of Recidivism") +  # Axis labels
    ggtitle(paste("Race:", race_i)) +  # Title with current race
    theme_minimal()  # Use a minimal theme for the plot
  
  # Store the plot in the list
  plots[[race_i]] <- p
}

# Print or display the plots
for (i in seq_along(plots)) {
  print(plots[[i]])
}

# Use the raw COMPAS risk scores to make a ROC curve for each race.

# Are the best discriminating models the ones you expected?
# Are the ROC curves helpful in evaluating the COMPAS risk score?

# Create a list to store the ROC curves
roc_curves <- list()

# Loop through each race category
for (race_i in race_categories) {
  # Subset the data for the current race
  race_data <- filter(risk, risk$race == race_i)
  # Create ROC curve for the current race
  roc_curve <- roc(outcome ~ score, data = race_data)
  # Store the ROC curve in the list
  roc_curves[[race_i]] <- roc_curve
}

# Plot the ROC curves for each race
for (i in seq_along(roc_curves)) {
  plot(roc_curves[[i]], main = paste("ROC Curve for Race:", race_categories[i]), col = i)
}

# Print or display the plots
for (i in seq_along(roc_curves)) {
  print(roc_curves[[i]])
}

# The data for asians and native americans make sense since the sample size is small.
# 
# As in part b, one thing that doesn't make a lot of sense to me is how my
# probability drops for risk scores of 10. It is possible that there's an error 
# in my code. If there isn't an error, this could be explained by errors in the 
# scoring system defaulting to a value of 10 for certain people or that individuals 
# with really high risk scores may tend to be more cautious of repeat offenses due to past experiences.
# 
# It does makes sense that the CI's grow with the risk score, since it's easier 
# (and more likely) to predict that someone  won't be a repeat offender rather 
# than that they will. 
# 
# It's interesting that the CI for african americans is the lowest at high risk
# scores. This make sense though since that group has a higher rate of recidivism.
# 
# Some datapoints show a probability of 1 or 0 or a number that doesn't align with
# the trend, but those are usually points that have a low (or even a single) data
# points for generating that probability for the group. 

# ROC Curves
# Use the raw COMPAS risk scores to make a ROC curve for each race. 
# 
# - Are the best discriminating models the ones you expected? 
#   - Are the ROC curves helpful in evaluating the COMPAS risk score? 


library(pROC)

# Create a list to store the ROC curves
roc_curves <- list()

# Loop through each race category
for (race_i in race_categories) {
  # Subset the data for the current race
  race_data <- filter(risk, risk$race == race_i)
  # Create ROC curve for the current race
  roc_curve <- roc(outcome ~ score, data = race_data)
  # Store the ROC curve in the list
  roc_curves[[race_i]] <- roc_curve
}

# Plot the ROC curves for each race
for (i in seq_along(roc_curves)) {
  plot(roc_curves[[i]], main = paste("ROC Curve for Race:", race_categories[i]), col = i)
}

# Print or display the plots
for (i in seq_along(roc_curves)) {
  print(roc_curves[[i]])
}

# Interpretation: 
#   
#   * Are the best discriminating models as expected?
#   The asian and native american curves show very high differentiation for 
# recidivism using the scores. The sample size for these groups are very small
# though, similar to as if the training data is very small for creating a model.
# Interestingly, the the AUCs for african american > caucasian > hispanic. This
# was not something I was expecting. Though it's not significantly different, the
#   scores seem to be the best at differentiating recidivism for african americans.
#   This makes sense though, because even though there are more caucasian datapoints,
#   african americans have a higher recidivism rate from the data given. The hispanic
#   ROC curve is also interesting, but again, there aren't a lot of datapoints for
# hispanics with higher risk scores. 
# Are the ROC curves helpful in evaluating the COMPAS risk score?
#   I feel like it is difficult to draw conclusions using the ROC curves because 
# the sample size, sample distribution for each race, and even judiciary
# prejudices affect so much of the data before it gets to creating a ROC curve.
# I feel like ROC curves can help justify decisions made elsewhere, but evaluating
# the risk scores based solely on the ROC curves seems a bit too presumptuous. 
