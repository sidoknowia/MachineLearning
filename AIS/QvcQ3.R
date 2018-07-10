setwd("~/git/MachineLearning/AIS")
library(psych)
graphics.off() 
par("mar") 
par(mar=c(1,1,1,1))
sh.data.read <- read.csv("transactionsq3.csv")
names(sh.data.read)

sh.data <- sh.data.read[,3:12]

sh.data <- na.omit(sh.data)

lr.model <- lm(data = sh.data, Total_Transactions~Mean_Business+Median_Business+Total_Business+Mean_Days+Median_Days+Mean_Initial_Delay+Median_Initial_Delay)

require(corrplot)
corrplot(cor(sh.data), method = c("number")) 




null <- lm(Total_Transactions~1, data=sh.data)
full <- lm(Total_Transactions~., data=sh.data) 
step(full, scope=list(lower=null, upper=full), direction="both")
step(null, scope=list(lower=null, upper=full), direction="both") 

lr.model2 <- lm(formula = Total_Transactions ~ Total_Business + Mean_Business + 
                  Median_Business + Median_Days + Mean_Days, data = sh.data)


lr.model3 <- lm(formula = Total_Transactions ~ Total_Business + Mean_Initial_Delay + Mean_Days, data = sh.data)

summary(lr.model2)
summary(lr.model3)

  par(mfrow=c(2,2))
plot(lr.model)