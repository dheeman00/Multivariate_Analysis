#install.packages(c('kernlab', 'mvtnorm', 'matrixcalc'), dependencies = TRUE)

library(matrixcalc)
library(mvtnorm)
library(kernlab)

#Dheeman Saha
#Final: STAT-715
#Date: 05/04/18

#Explanation:
#In this program we have trying to classifying 13 different pens that was provided 
# We have make use of different kernels: Correlation, Radial Based  and Polynomial 
# But Correlation outperformed the other so that's the oly kernel that is been used.
# Therefore, using that kernel we have initialized the 3 parameters
# Then created the new score matrix to test the performance of the classifier

#### DATA Loading ####

#load('~/Dropbox/Data_Science/STAT-715/Project/Project_STAT715/STAT715_ink_data_training.rData')

#include the location of the train data
load()
#include the train data
ink_data = ink.training.dat

#include the new dataset here
ink_data_test =
#ink_data_test = ink.training.dat
#ink_data_test = ink.training.dat[,,c(4:6), ]
#dim(ink_data_test)

#### Extracting Train/Test Data and PreProcessing ####

convert_data_vector <- function(ink_data)
{
  data_row = dim(ink_data)[1]
  data_col = dim(ink_data)[2]
  data_sample = dim(ink_data)[3]
  data_pen = dim(ink_data)[4]
  
  #creating the data size 
  data_size = data_row * data_col
  
  data_pen_total = array(data = NA, c(data_sample, data_size, data_pen))
  for (i in 1:data_pen) 
    {
      k <- 1
      data_mat = matrix(data = NA, nrow = data_row, ncol = data_col)
      for(j in 1:data_sample)
      {
        data_mat = as.matrix(ink_data[ , , j, i])
        data_pen_total[k, ,i] <- as.vector(data_mat)
        k <- k + 1
        data_mat <- matrix(data = NA, nrow = data_row, ncol = data_col)
      }
    }
  return(data_pen_total)
}

#information of each of the dimensions - train
data_row_train = dim(ink_data)[1]
data_col_train = dim(ink_data)[2]
data_sample_train = dim(ink_data)[3]
data_pen_train = dim(ink_data)[4]

#total train data in 3 dimension
data_total_train <- convert_data_vector(ink_data)
#dim(data_total_test)

# #information of each of the dimensions - test
# data_row_test = dim(ink_data_test)[1]
# data_col_test = dim(ink_data_test)[2]
# data_sample_test = dim(ink_data_test)[3]
# data_pen_test <- dim(ink_data_test)[4]

#total test data in 3 dimension
data_total_test <- convert_data_vector(ink_data_test)
#dim(data_total_test)


#### KERNEL TRAIN DATASET ####
#Used the correlation kernel cor()
score_pen = array(NA, dim = c(data_sample_train, data_sample_train, data_pen_train))
#dim(score_pen)

for (k in 1:data_pen_train) 
{
  row_1st = 1; row_2nd = 1  
  for (row_1st in 1:(data_sample_train)) 
  {
    for (row_2nd in 1 : data_sample_train) 
    {
      #here I have tested different kernels
      #Correlation
      score_pen[row_1st, row_2nd, k] = cor(data_total_train[row_1st, ,k] ,
                                           data_total_train[row_2nd, ,k])
      
      #Radial Based
      #rbfkernel <- rbfdot(sigma = 0.1)
      #score_pen[row_1st, row_2nd, k] = rbfkernel(data_total_train[row_1st, ,k], 
      #                                     data_total_train[row_2nd, ,k])
      
      #Polynomial
      #polykernel = polydot(degree = 1, scale = 1, offset = 1)
      #score_pen[row_1st, row_2nd, k] = polykernel(data_total_train[row_1st, ,k], 
      #                                         data_total_train[row_2nd, ,k])
      #                                         
      
    }  
  }
}


#checking +ve definite
# for(i in 1:dim(score_pen)[3])
# {
#   print(is.positive.definite(score_pen[,,i]))
# }


#### Creating the Design Matrix Training ####
#n_train = nrow(score_pen[,,1])
n_train = dim(score_pen)[1]
N_train = factorial(n_train) / (factorial(n_train-2) * factorial(2))

Design_Matix_Train = matrix(0, N_train, n_train)
design_counter <- 1
for (i in 1:(n_train - 1)) 
{
  for (j in (i + 1):n_train) 
  {
    Design_Matix_Train[design_counter, i] <- 1
    Design_Matix_Train[design_counter, j] <- 1
    design_counter <- design_counter + 1
  }
}


#Perform - P %*% t(P) - Training Part
P_Ptran_train <- Design_Matix_Train %*% t(Design_Matix_Train)
#dim(P_Ptran_train)

#### Calculation of the 3 parameters - using Training Dataset ####
#Calculation of Eigan Vector v1
v_1 = sqrt(1/N_train) * as.vector(matrix(1, nrow = N_train))

I_N_train = diag(N_train)

#Inner part of SSt
inner_part_SSt = I_N_train - (v_1 %*% t(v_1))

#taking the upper portion of the score value to avoid repeatation and diagonal values
score_reduced_train = matrix(NA, nrow = N_train, ncol = data_pen_train)
#dim(score_reduced_train)

counter <- 1
for (k in 1:data_pen_train) 
{
  counter <- 1
  for (i in 1: (nrow(score_pen)-1)) 
  {
    for (j in (i+1): nrow(score_pen)) 
    {
      score_reduced_train[counter, k] <- score_pen[i,j,k]  
      counter = counter + 1
    }  
  }  
}
#dim(score_reduced_train)

#Calculating SSt ref- equ 14
SS_t_total = as.vector(matrix(NA, nrow = data_pen_train))
#dim(score_reduced_train)
for (i in 1:data_pen_train) 
{
  SS_t_total[i] = t(score_reduced_train[,i]) %*% inner_part_SSt %*%(score_reduced_train[,i])
}

#Calculating s_bar
One_N_train = as.vector(matrix(1, nrow = N_train))

s_bar_total = as.vector(matrix(1, nrow = data_pen_train))
for (i in 1:data_pen_train) 
{
  s_bar_total[i] = (1/N_train) * t(One_N_train) %*% score_reduced_train[,i]  
}

#Calculating the grand mean of each of the pens
#grand_mean = matrix(NA, nrow = nrow(score_pen[1,,]), ncol = data_pen)
grand_mean = matrix(NA, nrow = dim(score_pen)[1], ncol = data_pen_train)
#dim(grand_mean)

for (i in 1:data_pen_train) 
{
  for (j in 1:dim(score_pen)[1]) 
  {
    grand_mean[j,i] <- s_bar_total[i]
  }
}

#Calculating (1/n-1)*t(P)*s_n
outer_part = 1/(n_train - 1) * t(Design_Matix_Train)
#dim(outer_part)
#dim(score_reduced_train)

local_mean = outer_part %*% score_reduced_train
#dim(local_mean)

#Calculating SSa
constant_part = ((n_train - 1)^2) / (n_train - 2)

#Calculating the difference between each pen mean and total pen mean
sum_difference_means = as.vector(matrix(NA, nrow = data_pen_train))
for (i in 1:ncol(grand_mean))
{
  sum_difference_means[i] = sum((local_mean[, i] - mean(grand_mean[,i]))^2) # 17 mean difference of each pen
}

#ref - eq.13 in the paper
#(n-1)^2/(n-2) sum(s_bar^(k) - s_bar)
SS_a_total = constant_part * sum_difference_means

#Calculating SSe
SS_e_total = SS_t_total - SS_a_total

#Calculating the mean square of SSa and SSe (ref- Table: 4)
MSS_a_total = SS_a_total / (n_train - 1)

MSS_e_total = SS_e_total / (N_train - n_train)

#Calculating the 3 parameters variance of a and e and mean
#(MS_a - MS_e) / (n-2)
variance_a = (MSS_a_total - MSS_e_total) / (n_train - 2)

#MS_e
variance_e = MSS_e_total

#s_bar
theta_hat_total = s_bar_total


#### Combining the training and testing dataset ####
# making use of the covariance matrix mention in page - 14
#dim(data_total_train)
#dim(data_total_test)

#information of each of the dimensions - test
data_row_test = dim(ink_data_test)[1]
data_col_test = dim(ink_data_test)[2]
data_sample_test = dim(ink_data_test)[3]
data_pen_test <- dim(ink_data_test)[4]

#creating the data size 
data_size_new = data_row_test * data_col_test

data_pen_total_new = array(data = NA, c((data_sample_train+data_sample_test), 
                                        data_size_new, 
                                        (data_pen_train*data_pen_test)))
#dim(data_pen_total_new)

#merging the test and train dataset to create the new dataset
k = 1
for (i in 1:data_pen_test) 
{
  for(j in 1:data_pen_train)
  {
    data_pen_total_new[ , , k] = rbind(data_total_train[ , , j], data_total_test[ , , i])
    k = k + 1
  }
}


#### KERNEL using new dataset ####
# Used the correlation kernel cor()
score_pen_new = array(NA, dim = c((data_sample_train+data_sample_test), 
                                  (data_sample_train+data_sample_test), 
                                  (data_pen_train*data_pen_test)))
#dim(score_pen_new)

for (k in 1:(data_pen_train*data_pen_test)) 
{
  row_1st = 1; row_2nd = 1  
  for (row_1st in 1:(data_sample_train+data_sample_test)) 
  {
    for (row_2nd in 1 : (data_sample_train+data_sample_test)) 
    {
      score_pen_new[row_1st, row_2nd, k] = cor(data_pen_total_new[row_1st, ,k] , 
                                               data_pen_total_new[row_2nd, ,k])
    }  
  }
}


#### Calculating the value of N and n - MERGE DATASET ####
n_test_new = dim(score_pen_new)[1]
#N_test_new = factorial(n_test_new) / (factorial(n_test_new - 2) * factorial(2))
N_test_new = choose(n_test_new,2)

#### Reduced Merged Dataset Score ####
#taking the upper portion of the score value to avoid repeatation
score_reduced_test_new = matrix(NA, nrow = N_test_new, ncol = (data_pen_train*data_pen_test))
#dim(score_pen_new)
#dim(score_reduced_test_new)
counter = 1
for (k in 1:(data_pen_train*data_pen_test)) 
{
  counter = 1
  for (i in 1: (nrow(score_pen_new)-1)) 
  {
    for (j in (i+1): nrow(score_pen_new)) 
    {
      score_reduced_test_new[counter, k] = score_pen_new[i,j,k]  
      counter = counter + 1
    }  
  }  
}

#### Creating the Design Matrix Merge Dataset ####

Design_Matix_Test_new = matrix(0, N_test_new, n_test_new)
#dim(Design_Matix_Test_new)
design_counter2 <- 1
for (i in 1:(n_test_new - 1)) 
{
  for (j in (i + 1):n_test_new) 
  {
    Design_Matix_Test_new[design_counter2, i] <- 1
    Design_Matix_Test_new[design_counter2, j] <- 1
    design_counter2 <- design_counter2 + 1
  }
}

#Perform - P %*% t(P) - Test Part
P_Ptran_test_new <- Design_Matix_Test_new %*% t(Design_Matix_Test_new)

#### Classifier ####

#initializing Vector of 1 N times
One_N_test_new = as.vector(matrix(1, nrow = N_test_new))

#initialize the diagonal matrix used for the sigma portion
I_N_test_new = diag(N_test_new)

#dim(score_reduced_test_new)[2]

#initialize the matrix to hold the classification of each pen
classification_pen = matrix(NA, nrow = data_pen_test, ncol = data_pen_test)

counter = 1
k = 1

#initialize the matrix that will hold the final value
classification_max_value = matrix(NA, nrow = data_pen_test)

for (j in 1:data_pen_test) #list of pens
{
  for (i in 1:data_pen_test) 
  {
    # refer eq-3 for mean and sigma inputs
    classification_pen[i,j] = dmvnorm(x = score_reduced_test_new[,k],
                                    mean = (theta_hat_total[j] %*% One_N_test_new),
                                    sigma = (P_Ptran_test_new * variance_a[j]) + (variance_e[j] * I_N_test_new)) 
    k = k +1
  }
  classification_max_value[j] = which.max(classification_pen[i,])
  print(paste(counter, classification_max_value[j]))
  counter = counter + 1
  
}
