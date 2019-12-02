options(repr.plot.width=5, repr.plot.height=4)
Packages <- c("dplyr","tidyr","readr","lubridate","ggplot2","tidyr","faraway","fastDummies",
              "glmnet","MASS","car","coefplot","plotmo","recipes","e1071","Metrics","anytime","caret")
#install.packages(Packages)
lapply(Packages, library, character.only = TRUE)
#DATA PREPROCESSING----
#Reading data from the file
data<-read.csv2("C:\\Users\\Student\\Thesis\\July_Week4.csv")
newfilename<-"July_Week4_Python.csv"
names(data)[1]<-"Streams"

#chart_date<-as.Date("2019-05-31")  #For week 1
#chart_date<-as.Date("2019-06-07") #For week 2
#chart_date<-as.Date("2019-06-14") #For week 3
chart_date<-as.Date("2019-06-21") #For week 4
data<-type.convert(data)
#Converting the dates to "time the date and chart_date"
data$Release_Date<-ceiling(difftime(strptime(chart_date,format = "%Y-%m-%d"),strptime(data$Release,format = "%d.%m.%Y"),units="weeks"))
data$Release_Date<-as.numeric(data$Release_Date)
#Summary of Release Dates
summary(data$Release_Date)
#Dividing Streams,duration and followers to 1000000 and 100000
data$Streams<-(data$Streams)/1000000
data$duration_ms<-as.numeric(as.character(data$duration_ms))
data$duration_ms<-(data$duration_ms)/100000
#Standard Scaling
data_scaled<-scale(subset(data,select = -c(key,mode,time_signature,Release_Date
                                           ,Genre,Artist._Type,Feat)),center = TRUE)
data_scaled<-data.frame(data_scaled)
data_scaled<-cbind.data.frame(data_scaled,data$key,data$mode,data$time_signature,data$Release_Date,
                              data$Genre,data$Artist._Type,data$Feat)

colnames(data_scaled)[15:21]<-c("key","mode","time_signature","Release_Date"
                                ,"Genre","Artist._Type","Feat")
#Dummy columns
data_scaled_matrix<-dummy_cols(data_scaled,
  select_columns = c("key","mode","time_signature","Genre","Artist._Type","Feat"))
#Deleting dummied columns
data_scaled_matrix<-subset(data_scaled_matrix, select = names(data_scaled_matrix) !="key")
data_scaled_matrix<-subset(data_scaled_matrix, select = names(data_scaled_matrix) !="mode")
data_scaled_matrix<-subset(data_scaled_matrix, select = names(data_scaled_matrix) !="time_signature")
data_scaled_matrix<-subset(data_scaled_matrix, select = names(data_scaled_matrix) !="Genre")
data_scaled_matrix<-subset(data_scaled_matrix, select = names(data_scaled_matrix) !="Artist._Type")
data_scaled_matrix<-subset(data_scaled_matrix, select = names(data_scaled_matrix) !="Feat")
#Selecting data points that only have Streams<=2
data_scaled_matrix<-subset(data_scaled_matrix,Streams<=2)
corr<-cor(data_scaled_matrix)
delete<-c("loudness","acousticness","Followers","Artist._Type_Male","Artist._Type_male","Artist._Type_Band",
          "key_3","time_signature_3","New_0","key_0","mode_0","time_signature_5",
          "Genre_alternative","Feat_0")
#Creating a copy of a data
data2<-data_scaled_matrix[, !names(data_scaled_matrix) %in% 
                            delete, drop = F]
#CALCULATING COSINE SIMILARITIES----
data2_yeoX<-as.matrix(data2[,2:9])
#Cosine similarity function
cosine_sim <- function(a, b) crossprod(a,b)/sqrt(crossprod(a)*crossprod(b))
similarity_matrix<-matrix(nrow = nrow(data2_yeoX), ncol = nrow(data2_yeoX))
for (i in 1:nrow(data2_yeoX)){
  for (j in 1:nrow(data2_yeoX)){
    similarity_matrix[i,j]<-cosine_sim(data2_yeoX[i,],data2_yeoX[j,])
  }
}
avg_simlr<-rowMeans(similarity_matrix)
avg_simlr<-scale(avg_simlr,center = TRUE)


avg_simlr<-as.data.frame(avg_simlr)
data2<-cbind.data.frame(data2[,1:9],avg_simlr,data2[,10:ncol(data2)])

colnames(data2)[10]<-"Similarity"

#YEO JOHNSON----
#Yeo-Johnson Transformation
yeo_data<-caret::preProcess(data2[,2:12],method=c("YeoJohnson"))
yeo_data_transformed<-predict(yeo_data,data2[,2:12],type=)
#Transformed data with dummies 
data_yeo<-cbind(data2$Streams,yeo_data_transformed,data2[,13:ncol(data2)])
colnames(data_yeo)[1]<-"Streams"
#Transforming response
transform<-powerTransform(data_yeo$Streams,family="bcnPower")
y_lambda<-round(transform$lambda)
y_gamma<-transform$gamma
data_final<-data_yeo
data_final$Streams<-bcnPower(data_final$Streams,lambda=y_lambda,gamma=y_gamma)

write_delim(data_final,newfilename,delim = ";")
