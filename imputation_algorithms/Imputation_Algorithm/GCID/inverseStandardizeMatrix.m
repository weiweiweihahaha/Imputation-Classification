function [originalData] = inverseStandardizeMatrix(datanew, meanVector, stdVector)  
    [sample, dimension] = size(datanew);  
    originalData = zeros(sample, dimension);  
    
    for i = 1:dimension  
        if ~isnan(stdVector(i)) 
            threshold = eps(class(datanew));
            originalData(:, i) = datanew(:, i) * stdVector(i) + meanVector(i);
            originalData(abs(originalData(:, i)) < threshold, i) = 0;
        else  
            originalData(:, i) = NaN;
        end  
    end  
end