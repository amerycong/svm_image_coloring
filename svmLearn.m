function svm = svmLearn(trainingData,trainingLabels,varargin)

disp('Training...')
numbers = unique(trainingLabels);
svm = cell(1,length(numbers));

for i = 1:length(numbers)
    number = numbers(i);
    labs = trainingLabels==number;
    svm{i} = fitcsvm(trainingData,labs,varargin{:});
end

end
