%TP Scores are 2 to 14192 = 14191 total training examples
%Prediction_Scores_TP = xlsread('data.xlsx', 'E2:N14192');
Prediction_Scores_TP = xlsread('data.xlsx', 'E2:N1001');
%TN Scores are 14193 to 36193 = 22001 training examples 

%Prediction_Scores_TN = xlsread('data.xlsx', 'E14193:N36193');
Prediction_Scores_TN = xlsread('data.xlsx', 'E14193:N15192');
Prediction_Scores = [Prediction_Scores_TP; Prediction_Scores_TN];

%just replace class label with a vector of 1's 
%class_TP = zeros(1, 14191)+1;
%class_TN = zeros(1, 22001)+0;

class_TP = zeros(1, 1000)+1;
class_TN = zeros(1, 1000)+0;


class =[class_TP class_TN];
[n m] = size(Prediction_Scores);
%Network Learning Rate:
lr = .005;
%Regularization Parameter (Set to 0 for no regularization)
reg_param = 0; 
n_hidden_nodes = 15;
rng(1);
W1 = rand(n_hidden_nodes, m);
W2 = rand(n_hidden_nodes, n_hidden_nodes);
W3 = rand(1, n_hidden_nodes);

% Interleave training examples
 [Prediction_Scores Indices] = shuffle(Prediction_Scores);
 class = class(Indices);
%%
count = 0;
nEpochs = 100;
err = zeros(1, nEpochs);
Cur_Error = 0;

Current_correct_1 = zeros(3, nEpochs);
Current_correct_2 = zeros(3, nEpochs);
Current_correct_3 = zeros(3, nEpochs);
Current_correct_TS = zeros(3, nEpochs);




for epoch = 1:nEpochs
    epoch
    for i = 1:n
        count = count+1;
        I = Prediction_Scores(i,:)';
        
        Z1 = W1*I;
        Y1 = sigmoid_DNN(Z1);
        Z2 = W2*Y1;
        Y2 = sigmoid_DNN(Z2);
        Z3 = W3*Y2;
        Y3 = sigmoid_DNN(Z3);
      

        %output layer error, use a cross entropy error function w/ sigmoid
        %activation 
        delta_o = (class(i) - Y3);
        %hidden layer error
        delta_h2 = Y2.*(1-Y2).*(W3.'*delta_o);
                
        delta_h1 =  Y1.*(1-Y1).*(W2.'*delta_h2);
        
        

        % Update the weights using regularization
         W3 = W3 + lr*delta_o*Y2'+lr*reg_param*W3 / n;
        W2 = W2 + lr*delta_h2*Y1'+lr*reg_param*W2 / n;
        W1 = W1 + lr*delta_h1*I'+lr*reg_param*W1 / n;
      
        %Calculate error at each epoch
        Cur_Error = Cur_Error + .5*(class(i) - Y3)^2;
        
    end
    
    err(epoch) = (1/n)*Cur_Error;
    Cur_Error = 0;
    
    %Check classification accuracy for each epoch for all test sets

       Current_correct_1(:,epoch) = validation(W1, W2, W3, 1)';
     %  Current_correct_2(:,epoch) = validation(W1, W2, W3,2)';
      %  Current_correct_3(:,epoch) = validation(W1, W2, W3,3)';

    


end
%     percent_v_regularization(1,y) = max(Current_correct_TS(1,:))
   % percent_v_regularization(2,y) = max(Current_correct_1(1,:));
%     percent_v_regularization(3,y) = max(Current_correct_2(1,:));
%     percent_v_regularization(4,y) = max(Current_correct_3(1,:));



%%

figure;
subplot(4, 1,1);
plot(err);
title('cross entropy err, reg param = 0, lr = .05');
legend('Total Rate', 'TPR', 'TNR');


subplot(4, 1, 2);
plot(Current_correct_1');
[m, I] = max(Current_correct_1');
str1=sprintf('%.2f', m(1));
str2=sprintf('%.2f', m(2));
str3=sprintf('%.2f', m(3));
str = {str1; str2; str3};
text(I,m,str);
str=sprintf('Percent Correct TS 1, Max Correct = %f, Epoch = %d',m(1),I(1) );
title(str)
legend('Total Rate', 'TPR', 'TNR');

subplot(4, 1, 3);
plot(Current_correct_2');
[m, I] = max(Current_correct_2');
str1=sprintf('%.2f', m(1));
str2=sprintf('%.2f', m(2));
str3=sprintf('%.2f', m(3));
str = {str1; str2; str3};
text(I,m,str);
str=sprintf('Percent Correct TS 2, Max Correct = %f, Epoch = %d',m(1),I(1) );
title(str)
legend('Total Rate', 'TPR', 'TNR');

subplot(4, 1, 4);
plot(Current_correct_3');
[m, I] = max(Current_correct_3');
str1=sprintf('%.2f', m(1));
str2=sprintf('%.2f', m(2));
str3=sprintf('%.2f', m(3));
str = {str1; str2; str3};
text(I,m,str);
str=sprintf('Percent Correct TS 3, Max Correct = %f, Epoch = %d',m(1),I(1) );
title(str)
legend('Total Rate', 'TPR', 'TNR');

%%
% Test the network
Prediction_Scores_Test_TP = xlsread('data.xlsx', 'E13193:N14192');
Prediction_Scores_Test_TN = xlsread('data.xlsx', 'E35194:N36193');
Prediction_Scores_Test = [Prediction_Scores_Test_TP; Prediction_Scores_Test_TN];

class_TP = ones(1, 1000);
class_TN = zeros(1, 1000);
class =[class_TP class_TN];
[n m] = size(Prediction_Scores_Test);
correctnum = 0;
outputs = zeros(1, n);
mislabeled = zeros(2, n);
miscnt = 1;
for i = 1:n
I = Prediction_Scores_Test(i,:)';
  
    Z1 = W1*I;
    Y1 = sigmoid_DNN(Z1);
    Z2 = W2*Y1;
    Y2 = sigmoid_DNN(Z2);
    Z3 = W3*Y2;
    Y3 = sigmoid_DNN(Z3);
  
    outputs(i) = Y3;
   if Y3 < .5
       label = 0;
   else
       label = 1;
   end
   if (label == class(i))
       correctnum = correctnum + 1;
   else
      
       mislabeled(1, miscnt) = Y3;
       mislabeled(2, miscnt) = class(i);
        miscnt = miscnt + 1;
   end
end
sprintf('Learning Rate = %.4f #Nodes = %d Epochs = %d', lr ,n_hidden_nodes, nEpochs)

Percent_Correct_Test_Set = (correctnum / n) * 100


%% Test set 1
Prediction_Scores_Test_TP = xlsread('data.xlsx','testing dataset I','E3:N122');
Prediction_Scores_Test_TN = xlsread('data.xlsx','testing dataset I', 'E123:N240');
Prediction_Scores_Test = [Prediction_Scores_Test_TP; Prediction_Scores_Test_TN];
correct_TP = 0;
correct_TN = 0;
class_TP = ones(1, 120);
class_TN = zeros(1, 118);
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
  
    outputs(i) = Y3;
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
Percent_Correct_Set_1 = (correctnum / n) * 100
TPR = correct_TP / size(Prediction_Scores_Test_TP,1) * 100
TNR = correct_TN / size(Prediction_Scores_Test_TN,1) * 100


%% Test set 2
Prediction_Scores_Test_TP = xlsread('data.xlsx','testing dataset II','E3:N6281');
Prediction_Scores_Test_TN = xlsread('data.xlsx','testing dataset II', 'E6283:N19522');
Prediction_Scores_Test = [Prediction_Scores_Test_TP; Prediction_Scores_Test_TN];
correct_TP = 0;
correct_TN = 0;
class_TP = ones(1, 6279);
class_TN = zeros(1, 13420);
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
  
    outputs(i) = Y3;
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
Percent_Correct_Set_2 = (correctnum / n) * 100
TPR = correct_TP / size(Prediction_Scores_Test_TP,1) * 100
TNR = correct_TN / size(Prediction_Scores_Test_TN,1) * 100
%% Test Set 3

Prediction_Scores_Test_TN = xlsread('data.xlsx','testing dataset III', 'E3:N7955');
Prediction_Scores_Test_TP= [];
Prediction_Scores_Test = [Prediction_Scores_Test_TN];

correct_TN = 0;
correct_TP = 0;

class_TN = zeros(1, 7953);
class =[class_TN];

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
  
    outputs(i) = Y3;
   if Y3 < .5
       label = 0;
   else
       label = 1;
   end
   if (label == class(i))
       correctnum = correctnum + 1;
       if label == 1
           correct_TP = correct_TP + 1;2
       else
           correct_TN = correct_TN + 1;
       end
   end
end
Percent_Correct_Set_3 = (correctnum / n) * 100
TPR = correct_TP / size(Prediction_Scores_Test_TP,1) * 100;
TNR = correct_TN / size(Prediction_Scores_Test_TN,1) * 100;

%%
save('smallset_lr_.005_reg_0_nodes_30.mat', 'W1', 'W2', 'W3', 'Current_correct_1', 'Current_correct_2', 'Current_correct_3');
