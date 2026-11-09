function X = DataCompletion(X, Method)

if ~any(isnan(X))
    error('The missing values should be marked as NaN in the input matrix.');
end

switch Method
    case 'EM'
        X = regem(X);
    otherwise
        error('Only KNN and EM is supported');
end