function runR2SNPipeline()
    current_dir = fileparts(mfilename('fullpath'));
    addpath(genpath(current_dir));
    %%% >>>>>>>>>>>>>>>>>>>>>>  main config <<<<<<<<<<<<<<<<<<<<<<<<< %%%
    %%% >>>>>>>>>>>>>>>>>>>>>>  main config <<<<<<<<<<<<<<<<<<<<<<<<< %%%
    output_path = 'path\to\R2SN';
    mri_paths_file = 'path\to\imglist.txt';
    atlas_path = fullfile(current_dir, 'template', 'BN_Atlas_246_1mm.nii');
    
    % Specify feature files
    feature_weighted = load(fullfile(current_dir, 'template', 'wi_90.mat'));
    feature_weighted = logical(feature_weighted.wi);
    %%% ============================================================ %%%
    
    % Turn off warnings
    warning('off', 'images:graycomatrix:scaledImageContainsNan');
    
    % Read MRI paths
    patternNii = dir(fullfile(mri_paths_file, '*.nii'));
    patternNiiGz = dir(fullfile(mri_paths_file, '*.nii.gz'));
    mri_paths = [fullfile({patternNii.folder}, {patternNii.name}), ...
                 fullfile({patternNiiGz.folder}, {patternNiiGz.name})]';
             
    % Get the list of already processed files
    output_dir = fullfile(output_path, 'R2SN');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir); % Create the output directory if it doesn't exist
    end
    processed_files = dir(fullfile(output_dir, '*.csv'));
    processed_names = cellfun(@(x) erase(x, '.csv'), {processed_files.name}, 'UniformOutput', false);
    
    % Filter out already processed files from mri_paths
    mri_names = cellfun(@(x) erase(x, {'.nii', '.nii.gz'}), mri_paths, 'UniformOutput', false);
    [~, mri_names_only, ~] = cellfun(@fileparts, mri_names, 'UniformOutput', false);
    to_process = ~ismember(mri_names_only, processed_names);
    mri_paths = mri_paths(to_process);
    
    % Check if mri_paths is empty
    if isempty(mri_paths)
        disp('No files to process. Exiting script.');
        return; % Exit the script
    end
    
    % Start parallel pool (if not already started)
    %numCores = 12;
    if isempty(gcp('nocreate'))
        %parpool(numCores);
        parpool;
    end
    
    % Calculate total number of tasks
    totalTasks = length(mri_paths);
    
    % Record the start time
    startTime = tic;
    
    % Use parfor for parallel execution
    parfor i = 1:totalTasks
        % Record the start time of the current task
        taskStartTime = tic;
        
        % Process the current task
        mri_path = strtrim(mri_paths{i});
        R2SNBuilder(mri_path, atlas_path, output_path, feature_weighted)
        
        % Record the end time of the current task
        taskElapsedTime = toc(taskStartTime);
        
        % Calculate the average time per task
        avgTimePerTask = toc(startTime) / i;
        
        % Estimate the remaining time
        remainingTasks = totalTasks - i;
        estimatedRemainingTime = remainingTasks * avgTimePerTask;
        
        % Display progress and estimated remaining time
        fprintf('Progress: %d/%d | Elapsed: %.2f sec | Avg: %.2f sec/task | Remaining: %.2f sec\n', ...
                i, totalTasks, toc(startTime), avgTimePerTask, estimatedRemainingTime);
    end
    
    % Close parallel pool
    delete(gcp('nocreate'));
end
