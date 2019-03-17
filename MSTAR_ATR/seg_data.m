% % % filepath = 'E:\study\À×´ïÊý¾Ý\mydata\train\2S1\';
% % % dirname = dir([filepath, '*.jpg'])
% % % dst_width = 100;
% % % dst_height = 100;
% % % for ii = 1:length(dirname)
% % %     filename_cur = dirname(ii,1).name;
% % %     img_cur = imread(strcat(filepath,filename_cur));
% % %     [width, height] = size(img_cur);
% % %     middle_x = floor(width/2);
% % %     middle_y = floor(height/2);
% % %     img_dst = img_cur(max(middle_x-dst_width/2,1):min(middle_x+dst_width/2,width),max(middle_y-dst_height/2,1):min(middle_y+dst_height/2,height));
% % %     
% % %     img_dst_path = strcat(filepath,filename_cur);
% % %     imwrite(img_dst,img_dst_path);
% % % end
function seg_data(filepath)
dirname = dir([filepath, '*.jpg'])
dst_width = 100;
dst_height = 100;
for ii = 1:length(dirname)
    ii
    filename_cur = dirname(ii,1).name;
    img_cur = imread(strcat(filepath,filename_cur));
    [width, height] = size(img_cur);
    middle_x = floor(width/2);
    middle_y = floor(height/2);
    img_dst = img_cur(max(middle_x-dst_width/2+1,1):min(middle_x+dst_width/2,width),max(middle_y-dst_height/2+1,1):min(middle_y+dst_height/2,height));
    
    img_dst_path = strcat(filepath,filename_cur);
    imwrite(img_dst,img_dst_path);
end



