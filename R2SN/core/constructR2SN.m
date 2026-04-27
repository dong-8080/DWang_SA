function R2SN = constructR2SN(features, wi)
    % features: 输入的特征表格 (table类型)
    % wi: 特征选择权重向量 (1表示选择该特征，0表示忽略)

    % 将table转换为数组形式以便处理
    features_array = table2array(features);
    
    % Step 1: 对每一列进行归一化处理
    normalized_features = normalize(features_array, 'range', [-1, 1]);
    
    % Step 2: 筛选选定的列
    selected_columns = wi; % 获取需要保留的列索引
    selected_normalized_features = normalized_features(:, selected_columns);
    
    % Step 3: 计算每个脑区之间的皮尔森相关性
    R2SN = corr(selected_normalized_features', 'Type', 'Pearson');
end