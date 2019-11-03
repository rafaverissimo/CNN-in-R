source("http://bioconductor.org/biocLite.R")
biocLite("EBImage")
library('EBImage')
library('keras')
library('magick') 
library('tensorflow')
library('reticulate')
reticulate::py_config()

#Treino
setwd('..\\')
library('readr')
traindata <- read_csv('train.csv') 
names_train <- traindata$id
trainy<-traindata$has_cactus
trainLabels <- to_categorical(trainy)

#Teste
testdata <- read_csv('sample_submission.csv') 
names_test <- testdata$id


#Imagens
setwd('..\\train')
train <- list()
for (i in 1:17500){
  train[[i]]<-readImage(names_train[i])
  train[[i]] <- resize(train[[i]], w = 32, h = 32)
}

setwd('..\\test')
test <- list()
for (i in 1:4000){
  test[[i]]<-readImage(names_test[i])
  test[[i]] <- resize(test[[i]], w = 32, h = 32)
}

#Arrumando
list2tensor <- function(xList) {
  xTensor <- simplify2array(xList)
  aperm(xTensor, c(4, 1, 2, 3))    
}

X_train <- list2tensor(train)
X_test <- list2tensor(test)

#Model
model <- keras_model_sequential()

model %>%
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(32, 32, 3)
  ) %>%
  layer_activation("elu") %>%
  layer_conv_2d(filter = 32,strides=1, kernel_size = c(3,3)) %>%
  layer_activation('elu') %>%
  layer_dropout(0.20) %>%
  layer_conv_2d(filter = 8,strides=2, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 24, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("elu") %>%
  layer_conv_2d(filter = 32,strides=1, kernel_size = c(3,3)) %>%
  layer_activation("elu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.20) %>%
  
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("elu") %>%
  layer_dropout(0.3) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

summary(model)

history <- model%>%
             fit(X_train,
                 trainy,
                 epochs=50,
                 batch_size=32,
                 validation_split=0.25,
                 verbose=1
                 )

pred <- model %>% predict_classes(X_test)
pred


submission <- data.frame(id = testdata$id, has_cactus = pred)

write_csv(submission, 'submission.csv')
