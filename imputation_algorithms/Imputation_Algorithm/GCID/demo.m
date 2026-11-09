function demo(dataset_name, name, inputPath, outputPath)
    currentPath = pwd;
    addpath(genpath(fullfile(currentPath, 'imputation')));
    addpath(genpath(fullfile(currentPath, 'ClusteringEvaluation')));
    addpath(genpath(fullfile(currentPath, 'auxiliary')));

    cur = fileparts(currentPath);
    cur = fileparts(cur);
    cur = fileparts(cur);

    if nargin < 3 || strlength(string(inputPath)) == 0
        dataset_parts = split(string(dataset_name), "_");
        dataset_dir = char(dataset_parts(1));
        inputPath = fullfile(cur, 'data', 'Missing_data', dataset_dir, strcat(dataset_name, '.csv'));
    else
        inputPath = char(string(inputPath));
    end
    if ~isfile(inputPath)
        error('GCID:MissingInput', 'Missing input dataset: %s', inputPath);
    end

    data = readmatrix(inputPath);
    sample = size(data, 1);
    dimension = size(data, 2);
    indexnum = 0;
    for i = 1:sample
        for j = 1:dimension
            if isnan(data(i, j))
                indexnum = indexnum + 1;
            end
        end
    end

    [missingindex, missing_numj] = checkindex(data, indexnum);
    [datanew, meanVector, stdVector] = standardizematrix(data);
    data_em = DataCompletion(datanew, 'EM');
    datanew = data_em;
    seed_value = str2double(getenv('FRAMEWORK_SEED'));
    if isnan(seed_value)
        seed_value = 1;
    end
    [~, centroids] = mykmeans(datanew, dimension, 'Replicates', 1, 'Start', 'sample', 'rdseed', seed_value);
    [X, ~, ~, ~] = gmm(datanew, centroids, missingindex, missing_numj, 500);
    originalData = inverseStandardizeMatrix(X, meanVector, stdVector);

    if nargin < 4 || strlength(string(outputPath)) == 0
        outputDir = fullfile(cur, 'data', 'Clean_data', name);
        if ~isfolder(outputDir)
            mkdir(outputDir);
        end
        outputPath = fullfile(outputDir, strcat(dataset_name, '_GCID_clean.csv'));
    else
        outputPath = char(string(outputPath));
        outputDir = fileparts(outputPath);
        if strlength(string(outputDir)) > 0 && ~isfolder(outputDir)
            mkdir(outputDir);
        end
    end

    writematrix(originalData, outputPath);
end
