function [imgs, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%
%   Note that this is based on the MATLAB code from 
%   Joan F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


	%see if there's a suffix, specifying one of multiple targets, for
	%example the dot and number in 'Jogging.1' or 'Jogging.2'.
% 	if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end))),
% 		suffix = video(end-1:end);  %remember the suffix
% 		video = video(1:end-2);  %remove it from the video name
% 	else
% 		suffix = '';
% 	end

	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\'
		base_path(end+1) = '\';
	end
	video_path = [base_path, video, '\'];
    
    % try to load initial position
    initial_filename = [video_path, 'init.txt'];
    f = fopen(initial_filename);
    assert(f ~= -1, ['No initial position to load ("' initial_filename '").']);
    % the format is [x, y, width, height]
    try
		intial = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
	catch  %#ok try different format (no commas)
		frewind(f);
		intial = textscan(f, '%f %f %f %f');  
    end
    intial = cat(2, intial{:});
	fclose(f);
    
    % set initial position and size
    % The ordering of coordinates and sizes is always [y, x].
	target_sz = [intial(1,4), intial(1,3)]; %[height,width]
	pos = [intial(1,2), intial(1,1)] + floor(target_sz/2);  %[y + height/2, x + width/2]

	% try to load ground truth from text file(Princeton's format)
    % ground truth may be missing for the EvaluationSet data
	groundtruth_filename = [video_path, video, '.txt'];
	f = fopen(groundtruth_filename);
    if f == -1
        disp('No ground truth to load');
    end
% 	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height]
	try
		ground_truth = textscan(f, '%f,%f,%f,%f,%f', 'Delimiter', ',','ReturnOnError',false);  
	catch  %#ok try different format (no commas)
		frewind(f);
		ground_truth = textscan(f, '%f,%f,%f,%f,%f');
	end
	ground_truth = cat(2, ground_truth{:});
	fclose(f);
	
	%store positions instead of boxes
    ground_truth = ground_truth(:, 1:4);
	ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
	
	%from now on, load all the RGB images and DEPTH images using
	%rgbdimgread
	imgs = rgbdimgread(video_path);
	
end

