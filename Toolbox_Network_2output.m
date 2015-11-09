%% Train a DNN to classify Missense Mutations using the DNN toolbox
clear all;
load Training_Data_2o.mat
load Classes_2o.mat
training_data = Prediction_Scores';
classes = class';

load TS1_2o.mat;
load TS1class_2o.mat;
TS1_data = Prediction_Scores_Test';
TS1_classes = class';

load TS2_2o.mat;
load TS2class_2o.mat;
TS2_data = Prediction_Scores_Test';
TS2_classes = class';


net = patternnet([25 10 ]);

% 
et.divideFcn = 'dividerand';
%net.trainParam.mc = 0.1;
net.divideParam.trainRatio = .9;
net.divideParam.valRatio = .1;
net.divideParam.testRatio = 0;
net.trainParam.max_fail = 200;
net.trainParam.lambda = 5e-7;
 net.layers{1}.transferFcn = 'logsig';
 net.layers{2}.transferFcn = 'logsig';
% 


net.trainParam.epochs = 30;
 net = train(net,training_data,classes);



TS1outputs = sim(net, TS1_data);

[FPR TPR t AUC_TS1] = perfcurve(TS1_classes(1,:), TS1outputs(1,:), 1);
%plotconfusion(TS1_classes,TS1outputs)
%plot(FPR, TPR)

correctnum = 0;
correct_TP = 0;
correct_TN = 0;
for i = 1:length(TS1outputs)
    if TS1outputs(1,i) < .5
       label = 0;
   else
       label = 1;
   end
   if (label == TS1_classes(1,i))
       correctnum = correctnum + 1;
       if label == 1
           correct_TP = correct_TP + 1;
       else
           correct_TN = correct_TN + 1;
       end
   end
end
Percent_Correct_Set_1 = (correctnum / length(TS1_classes(1,:))) * 100;
TPR1 = correct_TP / nnz(TS1_classes(1,:)) * 100;
TNR1 = correct_TN / (length(TS1_classes) - nnz(TS1_classes(1,:))) * 100;


TS2outputs = sim(net, TS2_data);
[FPR TPR t AUC_TS2, optop] = perfcurve(TS2_classes(2,:), TS2outputs(2,:), 1);

%y = net(test_data);

TS2outputs = sim(net, TS2_data);
[FPR TPR t AUC_TS2, optop] = perfcurve(TS2_classes(2,:), TS2outputs(2,:), 1);
%plotconfusion(TS2_classes,TS2outputs)

correctnum = 0;
correct_TP = 0;
correct_TN = 0;
for i = 1:length(TS2outputs)
    if TS2outputs(1,i) < .5
       label = 0;
   else
       label = 1;
   end
   if (label == TS2_classes(1,i))
       correctnum = correctnum + 1;
       if label == 1
           correct_TP = correct_TP + 1;
       else
           correct_TN = correct_TN + 1;
       end
   end
end
Percent_Correct_Set_2 = (correctnum / length(TS2_classes(1,:))) * 100;
TPR2 = correct_TP / nnz(TS2_classes(1,:)) * 100;
TNR2 = correct_TN / (length(TS2_classes(1,:)) - nnz(TS2_classes(1,:))) * 100;

%%
Test_Sets = {'TS 1'; 'TS 2'}
TotalAccuracy = [Percent_Correct_Set_1; Percent_Correct_Set_2];
TPR = [TPR1;TPR2];
TNR = [TNR1;TNR2];
AUC = [AUC_TS1; AUC_TS2];
T = table(TPR,TNR,AUC,TotalAccuracy, 'RowNames',Test_Sets);
T