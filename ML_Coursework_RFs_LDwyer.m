% Random Forest Model of UCI Cardiotocography dataset
% Student Name: Lesley Dwyer
% Data sourced from: https://archive.ics.uci.edu/ml/datasets/Cardiotocography 

% Load data into table
% Data includes 21 features, some unused columns and 1 class column (NSP)
cardio_data = readtable('CTG dataset.xls', 'ReadVariableNames', 1, 'Sheet', 'Data', 'Range', 'K2:AT2128');

% Remove unneeded columns
cardio_data = removevars(cardio_data, {'Var22', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'Var33', 'CLASS', 'Var35'});

% Get number of rows
n = size(cardio_data, 1);

% Look at mean, std dev, skew for each variable 
mean_cols = groupsummary(cardio_data, 'NSP', 'mean') % mean by class
std_dev_cols = groupsummary(cardio_data, 'NSP', 'std') % mean by class
cardio_data_matrix = table2array(cardio_data);
skew_cols = skewness(cardio_data_matrix) % skew for all columns

rng default % For reproducability

% Split data into 70% for training, 30% for testing 
divide_data = cvpartition(n, 'HoldOut', 0.3);
training_data = cardio_data(training(divide_data),:);
test_data = cardio_data(test(divide_data),:);

% Set hyperparameter values to check for grid search
% Try high enough values for trees to let error settle down
num_trees = [100, 250, 500]; 
% Try different number of predictors: 1 is mentioned in [4], 4 and 5 are suggested by [3] 
% and 21 is used to compare to bagging which uses all features
num_predictors = [1, 4, 5, 21]; 

% Initialise best parameters and error
best_num_trees = 100;
best_num_predictors = 1;
best_validation_error = 1;

% Grid search for hyperparameter tuning
for i = num_predictors 
    for j = num_trees
        % Check all combinations of hyperparameters on Random Forest model
        % Use Out of Bag error to find best combination
        model = TreeBagger(j, training_data, 'NSP', 'OOBPrediction', 'on', 'NumPredictorsToSample', i)
        OOB_error_all_trees = oobError(model);

        % Find Out of Bag error for current number of trees
        OOB_error = OOB_error_all_trees(j)

        % Save hyperparameters with the lowest error as best
        if OOB_error < best_validation_error
            best_num_predictors = i;
            best_num_trees = j;
            best_validation_error = OOB_error;
        % If error is the same, choose the one with the lower number of trees
        elseif (OOB_error == best_validation_error) & (best_num_trees > j)
            best_num_trees = j;
            best_num_predictors = i;
        end
    end
end

% Train the best model using best hyperparams and all training data
best_model = TreeBagger(best_num_trees, training_data, 'NSP', 'OOBPrediction', 'on', 'NumPredictorsToSample', best_num_predictors)

% Plot error against number of trees and predictors
best_OOB_Error = oobError(best_model);
training_error = best_OOB_Error(best_num_trees)
figure;
plot(best_OOB_Error, 'LineWidth', 1)
hold on;
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
title('Training Error - Best Num Trees and Predictors');
        
% Predict test data using best model 
predicted_NSP = str2double(predict(best_model, test_data));
 
% Show Confusion Matrix for test data
conf_matrix = confusionmat(test_data.NSP,predicted_NSP)
 
% Calculate Test Accuracy  
test_accuracy = (conf_matrix(1,1) + conf_matrix(2,2) + conf_matrix(3,3))/size(test_data, 1) * 100

% Calculate Test Precision, Recall, and F-Measure for each class

class = [1; 2; 3];
TP = [conf_matrix(1,1); conf_matrix(2,2);  conf_matrix(3,3)];
TN = [conf_matrix(2,2) + conf_matrix(2,3) + conf_matrix(3,2) + conf_matrix(3,3); 
    conf_matrix(1,1) + conf_matrix(1,3) + conf_matrix(3,1) + conf_matrix(3,3);
    conf_matrix(2,2) + conf_matrix(2,1) + conf_matrix(1,2) + conf_matrix(1,1)];
FP = [conf_matrix(2,1) + conf_matrix(3,1);
    conf_matrix(1,2) + conf_matrix(3,2);
    conf_matrix(2,3) + conf_matrix(1,3)];
FN = [conf_matrix(1,2) + conf_matrix(1,3);
    conf_matrix(2,1) + conf_matrix(2,3);
    conf_matrix(3,2) + conf_matrix(3,1)];
precision = [0;0;0];
recall = [0;0;0];
Fmeasure = [0;0;0];
test_performance = table(class, TP, TN, FP, FN, precision, recall, Fmeasure);
test_performance.precision(1) = test_performance.TP(1)/(test_performance.TP(1)+test_performance.FP(1))*100;
test_performance.precision(2) = test_performance.TP(2)/(test_performance.TP(2)+test_performance.FP(2))*100;
test_performance.precision(3) = test_performance.TP(3)/(test_performance.TP(3)+test_performance.FP(3))*100;
test_performance.recall(1) = test_performance.TP(1)/(test_performance.TP(1)+test_performance.FN(1))*100;
test_performance.recall(2) = test_performance.TP(2)/(test_performance.TP(2)+test_performance.FN(2))*100;
test_performance.recall(3) = test_performance.TP(3)/(test_performance.TP(3)+test_performance.FN(3))*100;
test_performance.Fmeasure(1) = 2*test_performance.recall(1)*test_performance.precision(1)/(test_performance.recall(1)+test_performance.precision(1));
test_performance.Fmeasure(2) = 2*test_performance.recall(2)*test_performance.precision(2)/(test_performance.recall(2)+test_performance.precision(2));
test_performance.Fmeasure(3) = 2*test_performance.recall(3)*test_performance.precision(3)/(test_performance.recall(3)+test_performance.precision(3));

% Summary of Test Performance
test_performance


