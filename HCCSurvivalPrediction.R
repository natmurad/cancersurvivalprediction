#title: "HCCDataSurvivalPrediction"
#author: "Natália Faraj Murad"
#date: "11/08/2021"

#Load Packages
library(data.table)
library(dplyr)
library(corrplot)
library(ggplot2)
library(gmodels)
library(caret) 
library(rpart)
library(rpart.plot)
library(glmnet)
library(arm)
library(mltools)

#Set Work Directory
#setwd("directory")

#Read Dataset
hccdata <- read.csv("hcc-data.csv", h = TRUE, sep = ",", na.strings = "?", dec = '.',
                    col.names = c('Gender', 'Symptoms', 'Alcohol',
                                  'HBsAg', 'HBeAg', 'HBcAb', 'HCVAb',
                                  'Cirrhosis', 'Endemic','Smoking',
                                  'Diabetes',	'Obesity','Hemochro',
                                  'AHT', 'CRI',	'HIV', 'NASH',
                                  'Varices', 'Spleno','PHT', 'PVT',
                                  'Metastasis',	 'Hallmark', 'Age',
                                  'Grams_day', 'Packs_year', 'PS',
                                  'Encephalopathy', 'Ascites', 'INR',
                                  'AFP', 'Hemoglobin', 'MCV', 'Leucocytes',
                                  'Platelets', 'Albumin', 'Total_Bil',
                                  'ALT', 'AST', 'GGT', 'ALP', 'TP',
                                  'Creatinine', 'Nodules','Major_Dim',
                                  'Dir_Bil', 'Iron', 'Sat', 'Ferritin',
                                  'Class'))
#Look at the Dataset Organization
#str(hccdata)
#summary(hccdata)
#dim(hccdata)

## Variables type
numeric.vars     <- c('Grams_day',	'Packs_year', 'INR',	'AFP',
                      'Hemoglobin', 'MCV',	'Leucocytes',	'Platelets', 
                      'Albumin',	'Total_Bil', 'ALT',	'AST',	'GGT',	'ALP',
                      'TP',	'Creatinine',	'Major_Dim',	'Dir_Bil',
                      'Iron',	'Sat',	'Ferritin')
categorical.vars <- c('Gender',	'Symptoms',	'Alcohol',	'HBsAg',
                      'HBcAb',	'HCVAb', 'Cirrhosis',	'Endemic',	'Smoking',
                      'Diabetes',	'Obesity',	'Hemochro', 'AHT',	'CRI',	
                      'HIV',	'NASH',	'Varices',	'Spleno',	'PHT',	'PVT',
                      'Metastasis',  'Hallmark', 'Class', 'PS', 'Encephalopathy', 'Ascites', 'Nodules')

ordinal.vars     <- c('PS', 'Encephalopathy', 'Ascites', 'Nodules')

#Check NAs
sapply(hccdata, function(x) sum(is.na(x)))

#Variables Distribution
cbind(freq=table(hccdata$Class), percentage=prop.table(table(hccdata$Class))*100)

#Density Distribution
par(mfrow=c(4,4))

for(i in 1:50) {
  plot(density(na.omit(hccdata[,i])), main=names(hccdata)[i])
}

#Boxplot
par(mfrow=c(1,4))
for(i in 1:50) {
  boxplot(na.omit(hccdata[,i]), main=names(hccdata)[i])
}

#Atributes by Class - Fazer por partes

jittered_x <- sapply(hccdata[,1:5], jitter)
pairs(jittered_x, names(hccdata[,1:5]), col=c('blue', 'red') )

par(mfrow=c(3,2))
for(i in 1:50) {
  barplot(table(hccdata$Class,hccdata[,i]), main=names(hccdata)[i],)
}

#Removing column: HBeAg(0:124, 1:1, NA:39)
hccdata$HBeAg <- NULL

#Smoking and Packs_Year

for(i in 1:nrow(hccdata)){
  if(is.na(hccdata$Packs_year[i]==TRUE)){
    hccdata$Packs_year[i] <- sample(na.omit(hccdata$Packs_year), 1)
  }
}

for(i in 1:nrow(hccdata)){
  if(hccdata$Packs_year[i]>0){
    hccdata$Smoking[i] <- 1
  } else {
    hccdata$Smoking[i] <- 0
  }
}

### Splitting classes
dead <- filter(hccdata, hccdata$Class=='1')
notdead <- filter(hccdata, hccdata$Class=='0')

#Other variables
for(i in 1:nrow(dead)){
  for(j in 1:ncol(dead)){
    if(is.na(dead[i,j]==TRUE)){
      dead[i,j] <- sample(na.omit(dead[,j]), 1)
    }  
  }
}

for(i in 1:nrow(notdead)){
  for(j in 1:ncol(notdead)){
    if(is.na(notdead[i,j]==TRUE)){
      notdead[i,j] <- sample(na.omit(notdead[,j]), 1)
    }  
  }
}

hccdata <- rbind(dead, notdead)
rm(dead, notdead)
invisible(gc())

## Formatting columns to factors function
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

## Factor variables
hccdata <- to.factors(df = hccdata, variables = categorical.vars)

datatabledata <- as.data.table(hccdata)
hccdata <- one_hot(datatabledata ,  cols = c('Encephalopathy', 'PS', 'Ascites', 'Nodules'), dropCols = TRUE)
hccdata <- as.data.frame(hccdata)

factorsvar <- c('Nodules_1', 'Nodules_2', 'Nodules_3', 'Nodules_4', 'Nodules_5', 'Ascites_1', 'Ascites_2', 'Ascites_3', 'Encephalopathy_1', 'Encephalopathy_2', 'Encephalopathy_3', 'PS_0', 'PS_1', 'PS_2', 'PS_3', 'PS_4')

hccdata <- to.factors(df = hccdata, variables = factorsvar)
rm(datatabledata)
invisible(gc())

### Age
hccdata$Age <- hccdata$Age/max(hccdata$Age)

# Normalization function 
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}
hccdata_scaled <- scale.features(hccdata, numeric.vars)

# Spliting train and test datasets
set.seed(123)
intrain<- createDataPartition(hccdata_scaled$Class,p=0.85,list=FALSE)
training<- hccdata_scaled[intrain,]
testing<- hccdata_scaled[-intrain,]
prop.table(table(testing$Class))
#class(training)
#training$Class
#View(training)

# Verifying correlation between numeric variables
numeric.var <- sapply(training, is.numeric) 
corr.matrix <- cor(training[,numeric.vars])  
corrplot::corrplot(corr.matrix, main="\n\n Correlation Between Numeric Variables", method="number")

# Feature Selection
training$Class <- as.factor(training$Class)
formula <- "Class ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
model <- train(formula, data = training, method = "bayesglm", trControl = control)
importance <- varImp(model, scale = FALSE)
#print(model)
# Plot
plot(importance)

#Function to select variables using Random Forest 
run.feature.selection <- function(num.iters=30, feature.vars, class.var){
  set.seed(10)
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var, 
                     sizes = variable.sizes, 
                     rfeControl = control)
  return(results.rfe)
}

rfe.results <- run.feature.selection(feature.vars = training[,-62], 
                                     class.var = training[,62])


#Visualization of results
#rfe.results

varImp((rfe.results))

# Train the Model
model <- "Class ~ ."
LogModel <- bayesglm(model, family=binomial,
                     data=training, drop.unused.levels = FALSE)

training$Class <- as.character(training$Class)
fitted.results <- predict(LogModel,newdata=training,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != training$Class)
print(paste('Logistic Regression Accuracy',1-misClasificError))


testing$Class <- as.character(testing$Class)
fitted.results <- predict(LogModel,newdata=testing,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != testing$Class)
print(paste('Logistic Regression Accuracy',1-misClasificError))


#Confusion Matrix of Logistic Regression
print("Confusion Matrix Para Logistic Regression")
table(testing$Class, fitted.results > 0.5)


# Train the Model
model <- "Class ~  Symptoms + Alcohol +       
HBsAg + HCVAb +            
 Smoking +          
Diabetes +          
AHT + CRI + HIV +             
NASH + Varices + Spleno +           
PHT + PVT +        
 Age +        
Packs_year +  PS_1 +             
 PS_3 + PS_4 +             
 Encephalopathy_2 + Encephalopathy_3 + 
Ascites_1 +  Ascites_3 +        
INR + AFP + Hemoglobin +       
MCV + Leucocytes + Platelets +        
Albumin + ALT +             
AST + ALP +             
TP + Creatinine +         
 Nodules_2 + Nodules_3 +        
  Major_Dim +        
Dir_Bil + Iron + Sat + 
Ferritin"
training$Class <- as.factor(training$Class)
LogModel <- bayesglm(model, family=binomial,
                     data=training, drop.unused.levels = FALSE)

training$Class <- as.character(training$Class)
fitted.results <- predict(LogModel,newdata=training,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != training$Class)
print(paste('Logistic Regression Accuracy',1-misClasificError))


testing$Class <- as.character(testing$Class)
fitted.results <- predict(LogModel,newdata=testing,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != testing$Class)
print(paste('Logistic Regression Accuracy',1-misClasificError))

#Confusion Matrix of Logistic Regression
print("Confusion Matrix Para Logistic Regression")
table(testing$Class, fitted.results > 0.5)