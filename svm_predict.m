%function [estimated_labels,performance,conf]=svm_predict(testing_data,testing_labels,training_labels,svm)
function [estimated_labels]=svm_predict(testing_data,training_labels,svm)
num_digits = size(testing_data,1);
numbers = unique(training_labels);
scores = zeros(num_digits,length(numbers));
for i = 1:length(numbers)
    number = numbers(i);
    [~,x0e,~]=predict(svm{i},testing_data);
    %how sure are you it is the number in question?
    scores(:,i) = x0e(:,2);
end

%get max label values
max_score_index = zeros(num_digits,1);
for i = 1:num_digits
    max_score_index(i,1)=find(scores(i,:)==max(scores(i,:)));
end

estimated_labels = numbers(max_score_index);

%performance = sum(estimated_labels==testing_labels)/num_digits;

%columns (across) is known labels, rows (down) is guessed labels
% conf = confusionmat(testing_labels,estimated_labels);
% disp('Confusion Matrix:')
% disp(conf)
% imagesc(conf)
% title([num2str(performance),' accuracy'])
% xlabel('True Digit')
% ylabel('Guessed Digit')

end
