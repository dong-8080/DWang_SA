library(mediation)
setwd("D:\\workspace")

results = list()
count = 1
stat_data = read.csv("protein_significant.csv")
data_all_proteins = read.csv("protein.csv")
for(row_idx in 1:nrow(stat_data)){
  for(col_idx in 1:ncol(stat_data)){
    protein_pvalue = stat_data[row_idx, col_idx]
    if (protein_pvalue<0.05){
      mediator_var = colnames(stat_data)[col_idx]
      treatment_vars = c("abeta", "ptau", "abeta")
      outcome_vars = c("ptau", "fdg", "fdg")
      for(item_id in c(1,2,3)){
        treatment_var = treatment_vars[item_id]
        outcome_var = outcome_vars[item_id]
        data = data_all_proteins[c(treatment_var, mediator_var, outcome_var, 'subtype')]
        data = na.omit(data)
        data = data[data$subtype==(row_idx-1), ]
        data_scaled = scale(data)
        data_scaled = as.data.frame(data_scaled)
        
        model_mediator_to_outcome = lm(as.formula(paste(outcome_var, "~", mediator_var)), data=data_scaled)
        model_mediator_to_outcome_results = summary(model_mediator_to_outcome)
        mediator_p_value = model_mediator_to_outcome_results$coefficients[mediator_var, 'Pr(>|t|)']
        
        if (mediator_p_value<0.05){
          model_treatment_to_mediator = lm(as.formula(paste(mediator_var, '~', treatment_var)), data = data_scaled)
          model_mediator_treatment_to_outcome = lm(as.formula(paste(outcome_var, '~', mediator_var, '+', treatment_var)), data = data_scaled)
          mediation_results = mediate(model_treatment_to_mediator, model_mediator_treatment_to_outcome, treat = treatment_var, mediator = mediator_var, boot = TRUE, sims = 10000)
          mediation_summary = summary(mediation_results)
          result = c(row_idx, treatment_var, mediator_var, outcome_var, 
                    mediation_summary$d.avg, mediation_summary$d.avg.p, 
                    mediation_summary$z.avg, mediation_summary$z.avg.p,
                    mediation_summary$tau.coef, mediation_summary$tau.p,
                    mediation_summary$n.avg, mediation_summary$n.avg.p)
          results[[count]] = result
          count = count+1  
        }
      }
    }
  }
}

df <- do.call(rbind, results)
colnames(df) <- c("ST", "Treatment", "Mediator", "Outcome", "ACME", "ACME_p", "ADE",  "ADE_p", "Total", "Total_p", "Prop", "Prop_p")
write.csv(df, "mediation_results.csv", row.names = FALSE )