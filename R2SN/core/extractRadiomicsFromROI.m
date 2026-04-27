function feature_vector = extractRadiomicsFromROI(t1_img, roi_mask)
    % 分组提取特征
    f1 = FirstOrderFeatures(t1_img, roi_mask); % 假设jt_1st_feature等函数可以处理单通道输入
    % f2 = SecondOrderFeatures(t1_img, roi_mask); % 这组特征不需要
    f3 = TextureFeatures(t1_img, roi_mask, 32, 16); % 参数可能需要根据实际情况调整
    
    % 提取f1中的特征
    f1_features = [
        f1.energy; 
        f1.kurtosis;
        f1.maximum;
        f1.mean; 
        f1.mad; 
        f1.median; 
        f1.minimum; 
        f1.range; 
        f1.rms; 
        f1.skewness; 
        f1.std; 
        f1.var; 
        f1.entropy; 
        f1.uniformity
    ];
    
    % 提取f3中的特征
    f3_features = [
        f3.Autocorrelation; 
        f3.ClusterProminence; 
        f3.ClusterShade; 
        f3.ClusterTendency; 
        f3.Contrast; 
        f3.Correlation; 
        f3.DifferenceEntropy; 
        f3.Dissimilarity; 
        f3.Energy; 
        f3.Entropy; 
        f3.Homogeneity1; 
        f3.Homogeneity2; 
        f3.IMC1; 
        f3.IMC2; 
        f3.IDMN; 
        f3.IDN; 
        f3.InverseVariance; 
        f3.MaximumProbability; 
        f3.SumAverage; 
        f3.SumEntropy; 
        f3.SumVariance; 
        f3.Variance; 
        f3.ShortRunEmphasis; 
        f3.LongRunEmphasis; 
        f3.GrayLevelNonuniformity; 
        f3.RunLengthNonuniformity; 
        f3.RunPercentage; 
        f3.LowGrayLevelRunEmphasis; 
        f3.HighGrayLevelRunEmphasis; 
        f3.ShortRunLowGrayLevelEmphasis; 
        f3.ShortRunHighGrayLevelEmphasis; 
        f3.LongRunLowGrayLevelEmphasis; 
        f3.LongRunHighGrayLevelEmphasis
    ];
    
    % 将f1和f3的特征合并为一个特征向量
    feature_vector = [f1_features; f3_features];
end