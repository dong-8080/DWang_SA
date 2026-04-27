function all_features = extractRadiomicsFromAtlas(t1img_path, atlas_path)
    % 读取影像和Atlas
    t1_img = load_nii(t1img_path);   % T1 MRI影像
    atlas_img = load_nii(atlas_path); % Atlas
    
    t1_img = t1_img.img;
    atlas_img = atlas_img.img;

    % 获取所有非零标签（即ROI编号）
    labels = unique(atlas_img);
    labels(labels == 0) = []; % 忽略背景标签（假设背景标号为0）
    
    feature_names = {'energy', 'kurtosis', 'maximum', 'mean', 'mad', 'median', ...
               'minimum', 'range', 'rms', 'skewness', 'std', 'var', ...
               'entropy', 'uniformity','Autocorrelation', 'ClusterProminence', ...
               'ClusterShade', 'ClusterTendency', 'Contrast', 'Correlation', ...
               'DifferenceEntropy', 'Dissimilarity', 'Energy', 'Entropy', ...
               'Homogeneity1', 'Homogeneity2', 'IMC1', 'IMC2', 'IDMN', ...
               'IDN', 'InverseVariance', 'MaximumProbability', 'SumAverage', ...
               'SumEntropy', 'SumVariance', 'Variance', 'ShortRunEmphasis', ...
               'LongRunEmphasis', 'GrayLevelNonuniformity', ...
               'RunLengthNonuniformity', 'RunPercentage', ...
               'LowGrayLevelRunEmphasis', 'HighGrayLevelRunEmphasis', ...
               'ShortRunLowGrayLevelEmphasis', 'ShortRunHighGrayLevelEmphasis', ...
               'LongRunLowGrayLevelEmphasis', 'LongRunHighGrayLevelEmphasis'};
           
    % table类型提取
    all_features = table('Size', [length(labels), length(feature_names)], ...
                     'VariableTypes', repmat({'double'}, 1, length(feature_names)), ...
                     'VariableNames', feature_names);

    % 遍历每个ROI编号
    for iLabel = 1:length(labels)
        label = labels(iLabel);

        % 根据当前label创建ROI mask
        roi_mask = (atlas_img == label);
        
        % 图像归一化
        t1_img = t1_img/max(t1_img(:));
        
        % 根据当前ROI mask计算裁剪范围
        [x_bounds, y_bounds, z_bounds] = computeBoundingBox(roi_mask);

        % 裁剪T1 MRI影像和ROI mask
        t1_img_cropped = t1_img(x_bounds(1):x_bounds(2), y_bounds(1):y_bounds(2), z_bounds(1):z_bounds(2));
        roi_mask_cropped = roi_mask(x_bounds(1):x_bounds(2), y_bounds(1):y_bounds(2), z_bounds(1):z_bounds(2));

        % 提取该label下的T1 MRI影像特征
        feature_vector = extractRadiomicsFromROI(t1_img_cropped, roi_mask_cropped);

%         % 存储结果
%         all_features(iLabel).label = label;
%         all_features(iLabel).features = features;
        % 存储结果到表格
        all_features{iLabel, :} = feature_vector';
    end
end