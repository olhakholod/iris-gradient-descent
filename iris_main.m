% view the data sets that are available in the Deep Learning Toolbox?  
help nndatasets

%% Preparing the Data

% load iris dataset
load iris_dataset

% load the input and target arrays into different names, such that...
% input corresponds to matrix X and the target corresponds to matrix T
[x,t] = iris_dataset;
size(x)
size(t)

%% Building the Neural Network Classifier

% set a random seed
setdemorandstream(491218382)

% set two-layer feed forward neural network with a single hidden layer...
% of 10 neurons  
net = patternnet(10);
view(net)

% training procedure
[net,tr] = train(net,x,t);
nntraintool

% performance on training, validation anf testing datasets
plotperform(tr)

%% Testing the Classifier

% test the training network with testing samples
% the network outputs will be in the range 0 to 1, ...
% so we can use vec2ind function to get the class indices as the position ...
% of the highest element in each output vector

testX = x(:,tr.testInd);
testT = t(:,tr.testInd);

testY = net(testX);
testIndices = vec2ind(testY)

% plot a confusion matrix
plotconfusion(testT,testY)

% overall percentages of correct and incorrect classification
[c,cm] = confusion(testT,testY)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

% perform ROC analysis
plotroc(testT,testY)
