function [x_bounds, y_bounds, z_bounds] = computeBoundingBox(img)
    % 计算三维影像中非零像素的最小边界，用于裁剪mask和图像。
    %
    % 输入:
    %   - img: 三维影像数据 (如MRI图像)
    %
    % 输出:
    %   - x_bounds: X轴方向上的裁剪范围 [xmin, xmax]
    %   - y_bounds: Y轴方向上的裁剪范围 [ymin, ymax]
    %   - z_bounds: Z轴方向上的裁剪范围 [zmin, zmax]
    
    % 计算Z轴方向的有效范围
    wi_z = sum(sum(img, 1), 2); % 对每个z层求和
    z_nonzero = find(wi_z);     
    z_bounds = [max(1, z_nonzero(1)-1), min(size(img, 3), z_nonzero(end)+1)];
    
    % 计算Y轴方向的有效范围
    wi_y = sum(sum(img, 3), 1); 
    y_nonzero = find(wi_y);     
    y_bounds = [max(1, y_nonzero(1)-1), min(size(img, 2), y_nonzero(end)+1)];
    
    % 计算X轴方向的有效范围
    wi_x = sum(sum(img, 3), 2); 
    x_nonzero = find(wi_x);     
    x_bounds = [max(1, x_nonzero(1)-1), min(size(img, 1), x_nonzero(end)+1)];
end