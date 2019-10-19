library(ff)

train <- read.csv(file = "~/projects/ashrae/train.csv", sep = ",", header = TRUE, nrows = 1000)
weather_train <- read.csv("~/projects/ashrae/weather_train.csv", header = TRUE, nrows = 1000)


print(head(train))
print(head(weather_train))


plot(weather_train$air_temperature, type = "l", lty = 1, col = "darkblue", ylim = c(-10, 40))
#plot(weather_train$cloud_coverage,type = "l", lty = 1, col = "darkblue")