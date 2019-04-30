clear
ReadPath = 'E:\study\雷达数据\MSTAR-PublicMixedTargets-CD1\MSTAR_PUBLIC_MIXED_TARGETS_CD1\45_DEG\COL2\SCENE1\ZSU_23_4\';
SavePath = 'E:\study\雷达数据\MSTAR-PublicMixedTargets-CD1\MSTAR_PUBLIC_MIXED_TARGETS_CD1\45_DEG\COL2\SCENE1\ZSU_23_4\';
FileType = '*.026';
Files = dir([ReadPath FileType]);
NumberOfFiles = length(Files);
for i = 1 : NumberOfFiles
    FileName = Files(i).name;
    NameLength = length(FileName);
    FID = fopen([ReadPath FileName],'rb','ieee-be');
    ImgColumns = 0;
    ImgRows = 0;
    while ~feof(FID)                                % 在PhoenixHeader找到图片尺寸大小
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'NumberOfColumns'))
            ImgColumns = str2double(Text(18:end));
            Text = fgetl(FID);
            ImgRows = str2double(Text(15:end));
            break;
        end
    end
    while ~feof(FID)                                 % 跳过PhoenixHeader
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'[EndofPhoenixHeader]'))
            break
        end
    end
    Mag = fread(FID,ImgColumns*ImgRows,'float32','ieee-be');
    Img = reshape(Mag,[ImgColumns ImgRows]);
    imwrite(((Img)),[SavePath FileName(1:NameLength-3) 'tif']); % 调整对比度后保存
    fclose(FID);
end

% % % function MSTAR2JPG(sourcePath, targetPath)
% % % if ~exist(targetPath,'dir')
% % %     mkdir(targetPath);
% % % end
% % % Files = dir(sourcePath);
% % % for i = 1:length(Files)
% % %     if Files(i).isdir == 0
% % %         FID = fopen([sourcePath '\' Files(i).name],'rb','ieee-be');
% % %         while ~feof(FID)                                % 在PhoenixHeader找到图片尺寸大小
% % %             Text = fgetl(FID);
% % %             if ~isempty(strfind(Text,'NumberOfColumns'))
% % %                 ImgColumns = str2double(Text(18:end));
% % %                 Text = fgetl(FID);
% % %                 ImgRows = str2double(Text(15:end));
% % %                 break;
% % %             end
% % %         end
% % %         while ~feof(FID)                                 % 跳过PhoenixHeader
% % %             Text = fgetl(FID);
% % %             if ~isempty(strfind(Text,'[EndofPhoenixHeader]'))
% % %                 break
% % %             end
% % %         end
% % %         Mag = fread(FID,ImgColumns*ImgRows,'float32','ieee-be');
% % %         Img = reshape(Mag,[ImgColumns ImgRows]);
% % %         imwrite(uint8(imadjust(Img)*255),[targetPath '\' Files(i).name(1:end-3) 'JPG']); % 调整对比度后保存
% % %         clear ImgColumns ImgRows
% % %         fclose(FID);
% % %     else
% % %         if strcmp(Files(i).name,'.') ~= 1 && strcmp(Files(i).name,'..') ~= 1
% % %             if ~exist([targetPath '\' Files(i).name],'dir')
% % %                 mkdir([targetPath '\' Files(i).name]);
% % %             end
% % %             MSTAR2JPG([sourcePath '\' Files(i).name],[targetPath '\' Files(i).name]);
% % %         end
% % %     end
% % % end
% % % end