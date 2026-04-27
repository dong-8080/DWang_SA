function R2SNBuilder(mri_path, atlas_path, output_path, wi)
    % mri_paths_file: ����MRIͼ��·����txt�ļ�·��
    % atlas_path: �̶���Atlas�ļ�·��
    % output_path: ���Ŀ¼·��, �Ƽ�ʹ��xx/dataset_name����ʽ�����ڹ���
    % wi: ����ѡ��Ȩ������ (1��ʾѡ���������0��ʾ����)

    % ������Ҫ����Ŀ¼
    radiomics_dir = fullfile(output_path, 'radiomics');
    if ~isdir(radiomics_dir)
        mkdir(radiomics_dir);
    end
    
    r2sn_dir = fullfile(output_path, 'R2SN');
    if ~isdir(r2sn_dir)
        mkdir(r2sn_dir);
    end

    % ������־�ļ�·��
    error_log_file = fullfile(output_path, 'error_log.txt');
    
    % ��ȡMRI�ļ���
    [~, name, ~] = fileparts(mri_path);
    if isempty(name)
        [~, name, ~] = fileparts(fullfile(mri_path)); % ��������·��
    end
    
    try
        % ��ȡ��MRI-Atlas�Ե�����
        features = extractRadiomicsFromAtlas(mri_path, atlas_path);
        
        % ��������CSV�ļ�
        output_csv = fullfile(radiomics_dir, [name, '.csv']);
        writetable(features, output_csv);
        
        %fprintf('Successfully extracted Radiomics features and saved at: %s\n', output_csv);

        % ����R2SN
        R2SN = constructR2SN(features, wi);
        
        % ����R2SN�����CSV�ļ�
        output_r2sn_csv = fullfile(r2sn_dir, [name, '.csv']);
        writematrix(R2SN, output_r2sn_csv, 'Delimiter', 'comma');
        
        %fprintf('Successfully construct R2SN and saved at: %s\n', output_r2sn_csv);
    catch ME
        % ʹ��fprintf���������Ϣ
        fprintf('Failed to process MRI file: %s. Error: %s\n', mri_path, ME.message);
        
        % ��¼������Ϣ��error_log.txt
        fid = fopen(error_log_file, 'a'); % 'a' ��ʾ׷��ģʽ
        if fid == -1
            error('�޷��򿪴�����־�ļ�����д��');
        end
        fprintf(fid, 'Failed to process MRI file: %s. Error: %s\n', mri_path, ME.message);
        fclose(fid);
    end

end