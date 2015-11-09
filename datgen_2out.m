%TP Scores are 2 to 14192 = 14191 total training examples
Prediction_Scores_TP = xlsread('data.xlsx', 'E2:N14192');
%TN Scores are 14193 to 36193 = 22001 training examples 
Prediction_Scores_TN = xlsread('data.xlsx', 'E14193:N28383');

Prediction_Scores = [Prediction_Scores_TP; Prediction_Scores_TN];

%just replace class label with a vector of 1's 
[num, class] = xlsread('data.xlsx', 'P2:P20');
%just replace class label with a vector of 1's for now;
class_TP = [ones(1, 14191); zeros(1,14191)];
class_TN = [zeros(1, 14191); ones(1,14191)];
class =[class_TP class_TN];

[n m] = size(Prediction_Scores);

% Interleave training examples
 [Prediction_Scores Indices] = shuffle(Prediction_Scores);
 class = class(:,Indices)';
 
 save('Training_Data_2o.mat', 'Prediction_Scores');
 save('Classes_2o.mat', 'class');
 
 %%
 
Prediction_Scores_Test_TP = xlsread('data.xlsx','testing dataset I','E3:N122');
Prediction_Scores_Test_TN = xlsread('data.xlsx','testing dataset I', 'E123:N240');
Prediction_Scores_Test = [Prediction_Scores_Test_TP; Prediction_Scores_Test_TN];

class_TP = [ones(1, 120); zeros(1, 120)];
class_TN = [zeros(1, 118); ones(1, 118)];
class =[class_TP class_TN]';
[n m] = size(Prediction_Scores_Test);
 save('TS1_2o.mat', 'Prediction_Scores_Test');
 save('TS1class_2o.mat', 'class');
 %%
Prediction_Scores_Test_TP = xlsread('data.xlsx','testing dataset II','E3:N6281');
Prediction_Scores_Test_TN = xlsread('data.xlsx','testing dataset II', 'E6283:N19522');
Prediction_Scores_Test = [Prediction_Scores_Test_TP; Prediction_Scores_Test_TN];

class_TP = [ones(1, 6279); zeros(1, 6279)];
class_TN = [zeros(1, 13240); ones(1, 13240)];
class =[class_TP class_TN]';
[n m] = size(Prediction_Scores_Test);


 save('TS2_2o.mat', 'Prediction_Scores_Test');
 save('TS2class_2o.mat', 'class');