# Load the necessary library
library(tidyverse)

data <-`Errors.(1).(1)`


# Reshape the data to a long format first
data_long <- data %>%
  pivot_longer(cols = X, names_to = "RMSE", values_to = "models")

# Pivot to wide format based on your need
data_wide <- data_long %>%
  pivot_wider(names_from = RMSE, values_from = models)

# Print the pivoted data
print("Pivoted data:")
print(data_wide)
getwd()
setwd("/Users/admin/Desktop")
# Save the wide data frame to a CSV file
write.csv(data_wide, "data_wide.csv", row.names = TRUE)





library(tidyr)
library(dplyr)



# Pivot the data to a wide format, calculating the mean for overlapping entries
df_wide <- data %>% 
  pivot_longer(cols = X, names_to = "RMSE", values_to = "models") %>%
  pivot_wider(names_from = RMSE, values_from = models, values_fn = list(Value = mean), values_fill = list(Value = NA))

# Print the wide dataframe
print(df_wide)




#######################################################
soyabean_data <- TG_ALLDATA_05_17_2021
str(soyabean_data)
summary(soyabean_data)
length(soyabean_data$YIELD)
sd(soyabean_data$YIELD)




library(tidyr)

# Example data (assuming the data is in wide format)
df_wide <- ANOVA_error

# Print original data
print("Original data:")
print(df_wide)

# Pivot data from wide to long format
df_long <- df_wide %>%
  pivot_longer(cols = everything(),   # Pivot all columns starting with "Model"
               names_to = "Model",            # Name the new column "Model"
               values_to = "Value")          # Name the values column "Value"

# Print long format data
print("Long format data:")
print(df_long)

anova <- aov(Value~Model, data = df_long)
summary(anova)
TukeyHSD(anova, conf.level = 0.95)



