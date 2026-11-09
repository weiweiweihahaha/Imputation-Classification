function [datanew, meanVector, stdVector] = standardizematrix(data)  
  
    [sample, dimension] = size(data);  
    datanew = data;  
    meanVector = zeros(1, dimension);  
    stdVector = zeros(1, dimension);  
    
    for i = 1:dimension  
        count = 0;  
        sum1 = 0;  
        sum2 = 0;  
        for j = 1:sample  
            if ~isnan(data(j,i))  
                count = count + 1;  
                sum1 = sum1 + data(j,i);  
                sum2 = sum2 + data(j,i)^2;   
            end  
        end  
        if count > 0
            meanVector(i) = sum1 / count;  
            stdVector(i) = sqrt((sum2 / count) - meanVector(i)^2);
            if stdVector(i) == 0
                stdVector(i) = NaN;
            end
        else  
            stdVector(i) = NaN;
        end  
        for m = 1:sample  
            if ~isnan(data(m,i))  
                datanew(m,i) = (data(m,i) - meanVector(i)) / stdVector(i);  
            end  
        end  
    end  
    
end