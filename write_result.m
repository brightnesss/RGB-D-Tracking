function [ flag ] = write_result( path, track_result, occ_results )
%WRITE_RESULT Write the tracking results to .txt on the format of the 
%Princeton.The format is :
%target_top_left_x,target_top_left_y,target_down_right_x,target_down_right_y(,target_state) newline
%target_top_left_x,target_top_left_y,target_down_right_x,target_down_right_y(,target_state) newline
%target_top_left_x,target_top_left_y,target_down_right_x,target_down_right_y(,target_state) newline
%...
%target_down_right_x = target_top_left_x + target_width
%target_down_right_y = target_top_left_y + target_height
%target_state is optional, indicating whether target is occluded, 1 if target is occluded, 0 otherwise. 
%If the target is not visible in a frame, all values should be "NaN"
%The name of this file should be the same with its sequence name. 


flag = false;

length = numel(occ_results);

track_result(:,3:4) = track_result(:,1:2) + track_result(:,3:4);

track_result = round(track_result);

id = find(occ_results);

for i = 1 : numel(id)
    track_result(id(i),:) = nan(1,4);
end

if exist(path,'file') == 2
    fid = fopen(path,'w');
else
    fid = fopen(path,'a');
end

for i = 1 : length
    fprintf(fid,'%d,%d,%d,%d,%d\n',track_result(i,1),track_result(i,2),...
        track_result(i,3),track_result(i,4),occ_results(i));
end

fclose(fid);

end

