# 0. SETUP					 		                                                        

# 0.1 - Clear existing workspace objects 
rm(list = ls())

# 0.2 - Set working directory to where the data file is located & results should be saved
setwd("/Users/adriellilopes/PycharmProjects/OB1-reader/contextual_semantic_similarity/data/")

# 0.3 - Load packages
library(lme4)
library(glue)
library(Matrix)
library(sjstats)
library(lmerTest)
library(dplyr)

# 1. Models

model<-"gpt2"
layer<-"11"
corpus<-"meco"
context_type<-"window"
type<-"saliency"
df<-read.csv(glue("processed/{corpus}/{model}/full_{model}_[{layer}]_{corpus}_{context_type}_{type}_df.csv"),header=T) 
head(df)
# dim(df)
# names(df)
# summary(df)

# 1.1. Intercept-only model and Random comparison

# 1.1.1. Total reading time
# Fixed intercept
IntOnly <- lm(dur ~ 1, data=df) 
summary(IntOnly)
AIC(IntOnly)
# Random intercept (1|uniform_id)
RandomIntPartOnly <- lmer(dur ~ 1 + (1|uniform_id), data=df)
summary(RandomIntPartOnly)
AIC(RandomIntPartOnly)
performance::icc(RandomIntPartOnly)
# Random intercept (1|trialid)
RandomIntTrialOnly <- lmer(dur ~ 1 + (1|trialid), data=df) 
summary(RandomIntTrialOnly)
AIC(RandomIntTrialOnly)
performance::icc(RandomIntTrialOnly)
# Random intercept (1|trialid) + (1|uniformid)
RandomIntOnly <- lmer(dur ~ 1 + (1|trialid) + (1|uniform_id), data=df) 
summary(RandomIntOnly)
AIC(RandomIntOnly)
performance::icc(RandomIntOnly)

# 1.1.2. Skipping
# Fixed intercept
IntOnly <- glm(skip ~ 1, data=df) 
summary(IntOnly)
AIC(IntOnly)
# Random intercept (1|uniform_id)
RandomIntPartOnly <- glmer(skip ~ 1 + (1|uniform_id), data=df, family="binomial")
summary(RandomIntPartOnly)
AIC(RandomIntPartOnly)
# Random intercept (1|trialid)
RandomIntTrialOnly <- glmer(skip ~ 1 + (1|trialid), data=df, family="binomial") 
summary(RandomIntTrialOnly)
AIC(RandomIntTrialOnly)

# 1.1.2. Rereading
# Fixed intercept
IntOnly <- glm(reread ~ 1, data=df) 
summary(IntOnly)
# Random intercept (1|uniform_id)
RandomIntPartOnly <- glmer(reread ~ 1 + (1|uniform_id), data=df, family="binomial")
summary(RandomIntPartOnly)
AIC(RandomIntPartOnly)
# Random intercept (1|itemid)
RandomIntTrialOnly <- glmer(reread ~ 1 + (1|trialid), data=df, family="binomial") 
summary(RandomIntTrialOnly)
AIC(RandomIntTrialOnly)


# 1.2. Mixed Models

# First Fixation Duration
model <- lmer(firstfix_dur ~ similarity + surprisal + frequency + length + (1|participant_id) + (1|trialid), data=df)
summary(model)

# Gaze Duration
model <- lmer(firstrun_dur ~ similarity + surprisal + frequency + length + (1|participant_id) + (1|trialid), data=df)
summary(model)

# Total Reading Time
model <- lmer(dur ~ similarity + surprisal + frequency + length + (1|participant_id) + (1|trialid), data=df)
summary(model)

# Skipping
model <- glmer(skip ~ similarity + surprisal + frequency + length + (1|participant_id) + (1|trialid), data=df, family="binomial")
summary(model)

# Rereading
model <- glmer(reread ~ similarity + surprisal + frequency + length + (1|participant_id) + (1|trialid), data=df, family="binomial")
summary(model)

# Next fixation chance (fixation data)
model <- glmer(sac_out_target ~ similarity + surprisal + frequency + length + sqrt(distance^2) + (1|participant_id), data=df, family="binomial")
summary(model)

# Check each direction separately
df_left <- df %>% filter(distance %in% c(-3,-2,-1))
model <- glmer(sac_out_target ~ similarity + surprisal + frequency + length + sqrt(distance^2) + (1|participant_id), data=df_left, family="binomial")
summary(model)
df_right <- df %>% filter(distance %in% c(3,2,1))
model <- glmer(sac_out_target ~ similarity + surprisal + frequency + length + sqrt(distance^2) + (1|participant_id), data=df_right, family="binomial")
summary(model)

# Check function vs content words
df_open <- df %>% filter(pos_tag %in% c('PROPN', 'VERB', 'NOUN', 'INTJ', 'ADV', 'ADJ'))
model <- glmer(sac_out_target ~ similarity + surprisal + frequency + length + sqrt(distance^2) + (1|participant_id), data=df_open, family="binomial")
summary(model)
df_closed <- df %>% filter(pos_tag %in% c('ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ'))
model <- glmer(sac_out_target ~ similarity + surprisal + frequency + length + sqrt(distance^2) + (1|participant_id), data=df_closed, family="binomial")
summary(model)

# Next fixation chance with saliency (fixation data)
model <- lmer(landing_target_position ~ pred_landing_target_position + target_surprisal + target_frequency + target_length + (1|participant_id), data=df)
summary(model)

# Check each direction separately
df_left <- df %>% filter(landing_target_position %in% c(-3,-2,-1))
model <- lmer(landing_target_position ~ pred_landing_target_position + target_surprisal + target_frequency + target_length + (1|participant_id), data=df_left)
summary(model)
df_right <- df %>% filter(landing_target_position %in% c(3,2,1))
model <- lmer(landing_target_position ~ pred_landing_target_position + target_surprisal + target_frequency + target_length + (1|participant_id), data=df_right)
summary(model)

# Predict with model
# predicted_regressionIn_likelihood <- predict(
#  glmerSaliencyRegInDist,
#  newdata = saliency_df_dist,
#  type = "response")
# saliency_df_dist$predicted_regressionIn_likelihood = predicted_regressionIn_likelihood
# write.csv(saliency_df_dist, glue("glmerSaliencyPlusTwoRegIn{distance}_predicted_regressions_{model}.csv"))
