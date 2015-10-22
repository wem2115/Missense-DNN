function accuracy = validation(W1, W2, W3,set)
%% Test set 1
if set == 0
    Prediction_Scores_Test_TP = xlsread('data.xlsx', 'E13193:N14192');
Prediction_Scores_Test_TN = xlsread('data.xlsx', 'E35194:N36193');
Prediction_Scores_Test = [Prediction_Scores_Test_TP; Prediction_Scores_Test_TN];
correct_TP = 0;
correct_TN = 0;
class_TP = ones(1, 1000);
class_TN = zeros(1, 1000);
else if set == 1
Prediction_Scores_Test_TP = xlsread('data.xlsx','testing dataset I','E3:N122');
Prediction_Scores_Test_TN = xlsread('data.xlsx','testing dataset I', 'E123:N240');
Prediction_Scores_Test = [Prediction_Scores_Test_TP; Prediction_Scores_Test_TN];
correct_TP = 0;
correct_TN = 0;
class_TP = ones(1, 120);
class_TN = zeros(1, 118);
else if set == 2
        Prediction_Scores_Test_TP = xlsread('data.xlsx','testing dataset II','E3:N1002');
        Prediction_Scores_Test_TN = xlsread('data.xlsx','testing dataset II', 'E6283:N7282');
        Prediction_Scores_Test = [Prediction_Scores_Test_TP; Prediction_Scores_Test_TN];
        correct_TP = 0;
        correct_TN = 0;
        class_TP = ones(1, 1000);
        class_TN = zeros(1, 1000);

    else
        Prediction_Scores_Test_TN = xlsread('data.xlsx','testing dataset III', 'E3:N7955');
        Prediction_Scores_Test_TP = [0];
Prediction_Scores_Test = [Prediction_Scores_Test_TN];
correct_TP = 0;
correct_TN = 0;
class_TP = [];
class_TN = zeros(1, 7953);
class =[class_TN];
    end
    end
end
class =[class_TP class_TN];
[n m] = size(Prediction_Scores_Test);
correctnum = 0;
outputs = zeros(1, n);

for i = 1:n
I = Prediction_Scores_Test(i,:)';
  
    Z1 = W1*I;
    Y1 = sigmoid_DNN(Z1);
    Z2 = W2*Y1;
    Y2 = sigmoid_DNN(Z2);
    Z3 = W3*Y2;
    Y3 = sigmoid_DNN(Z3);
  
    %outputs(i) = Y3;
   if Y3 < .5
       label = 0;
   else
       label = 1;
   end
   if (label == class(i))
       correctnum = correctnum + 1;
       if label == 1
           correct_TP = correct_TP + 1;
       else
           correct_TN = correct_TN + 1;
       end
   end
end
Percent_Correct = (correctnum / n) * 100;
TPR = correct_TP / size(Prediction_Scores_Test_TP,1) * 100;
TNR = correct_TN / size(Prediction_Scores_Test_TN,1) * 100;
accuracy = [Percent_Correct TPR TNR];
end