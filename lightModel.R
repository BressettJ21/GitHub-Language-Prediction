data <- read.csv('light_Model.csv')


x <-strsplit(data$tags_c,",")
y <- strsplit(data$Text_Cleaned, ",")
for (i in 1:685){
  data[i,'num_tags'] <- length(x[[i]])
  data[i, 'num_words'] <- length(y[[i]])
}



smallData <- data[c(-1,-2,-4,-7)]
smallData$License <- as.factor(smallData$License)


