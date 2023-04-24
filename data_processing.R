# setup
library(tidyverse)
library(tidyr)
library(dplyr)
library(fastDummies)
library(caret)

#####################
## Reading in Data ##
#####################

# to download raw dataset, navigate to https://www.icpsr.umich.edu/web/NACDA/studies/4690/datadocumentation#
# click on download, then select R

# read in data (note that "da04690.0001" is the name of the provided dataset)
# change this name if the .rda file has a different naming convention

# set to appropriate file path 
load("04690-0001-Data.rda")
# set appropriate name for rda data object
df <- as.data.frame(da04690.0001)

###############################
## Manual Variable Selection ##
###############################

# define columns we would like to keep (initial list was manually selected)
cols.to.keep<-c('V1','V101','V201','V202','V219','V220','V221','V222','V223','V224','V225','V301','V312','V318','V319','V320','V321','V322','V323','V324','V326','V328','V329','V330','V331','V332','V333','V334','V335','V402','V409','V420','V425','V440','V441','V503','V514','V527','V534','V535','V536','V537','V538','V539','V545','V811','V812','V813','V814','V819','V820','V821','V822','V823','V828','V829','V830','V831','V832','V833','V834','V835','V836','V837','V838','V839','V841','V901','V902','V914','V915','V916','V924','V928','V930','V932','V934','V936','V942','V943','V946','V947','V1002','V1023','V1024','V1025','V1026','V1027','V1028','V1029','V1030','V1031','V1032','V1033','V1034','V1101','V1110','V1111','V1116','V1117','V1148','V1301','V1302','V1303','V1304','V1309','V1310','V1311','V1312','V1313','V1326','V1327','V1328','V1329','V1501','V1505','V1509','V1513','V1520','V1525','V1601','V1602','V1603','V1604','V1605','V1606','V1607','V1608','V1609','V1610','V1802','V2002','V2003','V2007','V2011','V2015','V2016','V2017','V2020','V2060','V2061','V2062','V2063','V2203','V2219','V2306','V2307','V2308','V2309','V2310','V2311','V2313','V2314','V2400','V2403','V2404','V2405','V2600','V2601','V2603','V2604','V2605','V2606','V2607','V2609','V2611','V2612','V2613','V2614','V2615','V2616','V2617','V2618','V2619','V2621','V2622','V2623','V2625','V2631','V2650','V2800','V2801','V2802','V2803','V2804','V2805','V2806','V3000','V3002','V3003','V3013','V3032','V3033','V3200','V3203','V3204','V3205','V3206','V3401','V3407')

df<-df%>%select(all_of(cols.to.keep))

# read in variable names
var.names<-attributes(da04690.0001)$names
# read in variable labels
var.labels<-attributes(da04690.0001)$variable.labels
# add variable labels as names to our list
names(var.labels)<-var.names

# create list of the column labels for the columns we kept
col.names<-c()
for(i in cols.to.keep){
  col.names<-c(col.names,var.labels[[i]])
}

colnames(df)<-col.names

####################
# Data proccessing #
####################

## our data processing focusses on dealing with categorical data
## many ordinal fields will be treated as numeric

# create copy of original dataframe for processing
df.processed<-df

# create list of all variables which have type factor
factor.vars<-c()
for(c in colnames(df)){
  if(class(df[,c])=="factor"){
    factor.vars<-c(factor.vars,c)
  }
}

# remove variables from list for which we will need custom processing
cust.vars<-c('C6:CKPT-EVER/NEVER MARRY','H5:SURE WORKOUT AS WANTD','H6:CARRYOUT PLANS/CHANGE','IMPUTED FAMILY INCOME')

# removing variables which are in the custom list
factor.vars<-factor.vars[!(factor.vars %in% cust.vars)]

## to process the factor variables we use the following logic
# 1. Check the number of factor levels. If there are 2 levels, move on to step 2. If there are >2 levels, move on to step 3
# 2. For 2 level factors, dummy coding is used. Factor level beginning with "(1)" is coded as 1, and "(5)" is coded as 0
# 3. For >2 level factors, we code the number in parantheses as the numeric values, e.g. "(3)" is coded as 3

for(f in factor.vars){
  # check if there are 2 levels
  if(nlevels(df[,f])==2){
    # dummy encode 'yes' 'no' variables
    df.processed[,f] = case_when(
      grepl("(1)", df[,f])~ 1,
      grepl("(5)", df[,f])~0,
      grepl("(0)",df[,f])~0,
      grepl("(2)",df[,f])~0)
  }
  else{
    # pull out the numeric value for the ordinal responses
    df.processed[,f] = as.integer(substr(df[,f],2,2))
  }
}

# custom logic for the remaining variables
for(c in cust.vars){
  if(nlevels(df.processed[,c])==2){
    # dummy encode variables which have only 2 possible values
    df.processed[,c] = case_when(
      grepl("(1)", df.processed[,c]) ~ 1,
      grepl("(2)", df.processed[,c]) ~ 0)
  }
  # create a continuous variable to represent ordinal income
  else if(c == "IMPUTED FAMILY INCOME"){
    df.processed[,c] = as.integer(substr(df.processed[,c],2,2))
  }
}

##########################
# Deal with missing data #
##########################

# note that some data is considered missing at random
# whereas other data is missing due to an answer to a previous survey question (missing by design)
# these variables will be treated in custom ways

# function which replaces NA values in a column with a specified value
# returns the column with replaced NA values

replace_na<-function(col,replacement,df){
  df[is.na(df[,col]),col]<-replacement
  df[,col]
}

# B11:MOVE RETIRMNT COMMUN
# if value is equal to 6 (already live in retirement community), change to 0 to reflect the ordinal nature of the data
# fill in missing data with the middle of the ordinal range, 3
df.processed$`B11:MOVE RETIRMNT COMMUN`<-ifelse(df.processed$`B11:MOVE RETIRMNT COMMUN`==6,0,df.processed$`B11:MOVE RETIRMNT COMMUN`)
df.processed$`B11:MOVE RETIRMNT COMMUN`<- df.processed$`B11:MOVE RETIRMNT COMMUN` %>% 
  replace(is.na(.), 3)

# C1A:LIVE IN INTIMATE REL
# missing data here corresponds to married, so fill in with 1 to create a variable which includes married as living in an intimate relationship
df.processed[,"C1A:LIVE IN INTIMATE REL"]<-replace_na("C1A:LIVE IN INTIMATE REL",1,df.processed)

# list of columns where NA can be replaced by 0
l<- c("W1.White-Collar=1,0=Other", "W1.Blue-Collar=1,0=Other","C46D:DELT W/DEATH PARENT","C25:MOM ALIVE-OTHER MOM","C23A:#CHILDREN WHO DIED","C11:# TIMES WIDOWED", "C12:NUMBER OF DIVORCES","C1A:LIVE IN INTIMATE REL","RJ14:OTH BETOFF-WRK,1=NO","RJ13:R BETTROFF WRK,1=NO","SPOUSE IN HH?","J22:SER PROB WORK-3YEARS","J10(DOL):AMT EARNED-DOLR","J9:#HOURS WORK PER WEEK","J3:#WKS EMPLOYED-12 MOS","J2:ANY WORK FOR PAY NOW","G28A:#DAYS DRANK LAST MO","PARENTAL CHRONIC STRESS","MARITAL HARMONY, MEAN","DEPENDENCY ON SPOUS,MEAN","NEGATV SPOUSE BEHAV, SUM","RG8:DIF CLIMB STAIR 1=NO","RG11:DIF HVY HSEWRK 1=NO")
# fill in NAs in these columns with 0's
for(col in l){
  df.processed[,col]<-replace_na(col,0,df.processed)
}

##################################
# Drop rows with missing outcome #
##################################

# we will use a missing at random assumption to drop rows where our outcome of interest
# "B1:SATISFACTION W/LIFE" is missing

df.processed<-df.processed[!is.na(df.processed[,"B1:SATISFACTION W/LIFE"]),]


###########################################################
# Fill in remaining NAs (assumed MAR) with knn imputation #
###########################################################

preProc<-preProcess(df.processed,method="knnImpute")
# create dataframe of imputed values
imp_df<-predict(preProc,df.processed)

# loop through columns and fill in missing values with the knn imputed values

for (c in colnames(df.processed)){
  df.processed[is.na(df.processed[,c]),c]<-imp_df[is.na(df.processed[,c]),c]
}

###################################
# Write clean dataset to csv file #
###################################

write.csv(df.processed,file='processed_data.csv',row.names = F)

