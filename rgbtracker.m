function [positions, rect_results, time, occ_results] = tracker(video_path, rgbdimgs, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to 
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014
%
%   revised by: Yang Li, August, 2014
%   http://ihpdep.github.io


addpath('./utility');
temp = load('w2crs');
w2c = temp.w2crs;

%if the target is large, lower the resolution, we don't need that much
%detail
resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
if resize_image
    pos = floor(pos / 2);
    target_sz = floor(target_sz / 2);
end
target_sz_back = target_sz;

%window size, taking padding into account
window_sz = floor(target_sz * (1 + padding));
current_size =1;
% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);

	
%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

%store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf,2))';

search_size = [1  0.985 0.99 0.995 1.005 1.01 1.015];%
% search_size = [1, 0.99, 1.01,];
if show_visualization  %create video interface
    update_visualization = show_video(rgbdimgs.rgb, video_path, resize_image);
end
	
	
%note: variables ending with 'f' are in the Fourier domain.
length = numel(rgbdimgs.rgb);
time = 0;  %to calculate FPS
positions = zeros(length, 2);  %to calculate precision
rect_results = zeros(length, 4);  %to calculate
occ_results = zeros(length,1);
response = zeros(size(cos_window,1),size(cos_window,2),size(search_size,2));
szid = 1;
param1 = zeros(size(search_size,2), 6);

occ_max = 0;
APCE = 0;
occ_model_max = 0;
occ_model_APCE = 0;
occ_flag = false;   % whether the target is occluded
occ_max_lambda1 = 0.4;  % occ_flag from false to true
occ_max_lambda2 = 0.3;  % occ_flag from true to false
occ_APCE_lambda1 = 0.4;  % occ_flag from false to true
occ_APCE_lambda2 = 0.3;  % occ_flag from true to false
occ_update_factor = 0.9;  % update factor for exp average algorithm
    
    
for frame = 1 : length
    %load image
    rgbim = rgbdimgs.rgb{frame};
    depthim = rgbdimgs.depth{frame};
    if resize_image
        rgbim = imresize(rgbim, 0.5);
        depthim = imresize(depthim, 0.5);
    end
    
%     t0 = clock;
%     depth_time = 0;
    
    tic;
    
    if frame > 1
        %obtain a subwindow for detection at the position from last
        %frame, and convert to Fourier domain (its size is unchanged)
        %patch = get_subwindow(rgbim, pos, window_sz);
        %在rgb通道上处理尺度变换，融合rgb的结果与depth的结果
        
        %先在rgb通道上处理尺度变换
        for i=1:size(search_size,2)
            tmp_sz = floor((target_sz * (1 + padding)) * search_size(i));
            param1(i,:) = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
                tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
            param1(i,:) = affparam2mat(param1(i,:));
            patch_rgb = uint8(warpimg(double(rgbim), param1(i,:), window_sz));
            patch_depth = uint8(warpimg(double(depthim), param1(i,:), window_sz));
            x_rgb = get_features(patch_rgb, features, cell_size, cos_window,w2c);
            x_depth = get_features(patch_depth, features, cell_size, cos_window,w2c);
            x = x_rgb;
            x(:,:,end+1:end+size(x_depth,3)) = x_depth;
            zf = fft2(x);
            
            %calculate response of the classifier at all shifts
            switch kernel.type
                case 'gaussian'
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                case 'polynomial'
                    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                case 'linear'
                    kzf = linear_correlation(zf, model_xf);
            end
            response(:,:,i) = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
        end % end of for i=1:size(search_size,2)
        
        %target location is at the maximum response. we must take into
        %account the fact that, if the target doesn't move, the peak
        %will appear at the top-left corner, not at the center (this is
        %discussed in the paper). the responses wrap around cyclically.
        [~,tmp, ~] = find(response == max(response(:)), 1);

        szid = floor((tmp-1)/(size(cos_window,2)))+1;

        maxresponse = response(:,:,szid);
        
        % calculate occlusion paramter
        %max response of rgb respnse map
        occ_max = max(maxresponse(:));
        
        %average peak-to-correlation energy(APCE)
        APCE = (max(maxresponse(:)) - min(maxresponse(:))) ^ 2 ...
            / mean(mean((maxresponse - min(maxresponse(:))) .^ 2));
        
        % for test the response map
%         if frame >= 60
%             f2 = figure(2);
%             imshow(fftshift(response(:,:,szid)));
%             set(f2,'position',[300,300,600,600]);
%             f3 = figure(3);
%             imshow(fftshift(depth_response));
%             set(f3,'position',[700,300,600,600]);
%             f4 = figure(4);
%             imshow(fftshift(final_response));
%             set(f4,'position',[900,300,600,600]);
%             pause;
%         end
        
        % if occ_flag = true, do not need to update pos and target
        if ~occ_flag
            
            %target location is at the maximum response. we must take into
            %account the fact that, if the target doesn't move, the peak
            %will appear at the top-left corner, not at the center (this is
            %discussed in the paper). the responses wrap around cyclically.
            [vert_delta, horiz_delta] = find(maxresponse == max(maxresponse(:)), 1);
            if vert_delta > size(zf,1) / 2  %wrap around to negative half-space of vertical axis
                vert_delta = vert_delta - size(zf,1);
            end
            
            if horiz_delta > size(zf,2) / 2  %same for horizontal axis
                horiz_delta = horiz_delta - size(zf,2);
            end
            
            target_sz = target_sz * search_size(szid);
            tmp_sz = floor((target_sz * (1 + padding)));
            current_size = tmp_sz(2)/window_sz(2);
            pos = pos + current_size * cell_size * [vert_delta - 1, horiz_delta - 1]; %新的位置
        end % end of if occ_flag
    end % end of if frame > 1
    
    %obtain a subwindow for training at newly estimated target position
    
    
    if ~occ_flag
        if frame ~= 1
            %             target_sz = target_sz * search_size(szid);
            %             tmp_sz = floor((target_sz * (1 + padding)));
            %             param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
            %                     tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
            %             param0 = affparam2mat(param0);
            param0 = param1(szid,:);
        else
            target_sz = target_sz * search_size(szid);
            tmp_sz = floor((target_sz * (1 + padding)));
            param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
                tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
            param0 = affparam2mat(param0);
        end
        
        patch_rgb = uint8(warpimg(double(rgbim), param0, window_sz));
        patch_depth = uint8(warpimg(double(depthim), param0, window_sz));
        %为了统一hog和cn的维度，以hog的维度为准，将patch降维到hog特征维度
        x_rgb = get_features(patch_rgb, features, cell_size, cos_window,w2c);
        x_depth = get_features(patch_depth, features, cell_size, cos_window,w2c);
        x = x_rgb;
        x(:,:,end+1:end+size(x_depth,3)) = x_depth;
        xf = fft2(x);
        
        %Kernel Ridge Regression, calculate alphas (in Fourier domain)
        switch kernel.type
            case 'gaussian'
                kf = gaussian_correlation(xf, xf, kernel.sigma);
            case 'polynomial'
                kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
            case 'linear'
                kf = linear_correlation(xf, xf);
        end
        alphaf = yf ./ (kf + lambda);   %equation for fast training
        
    end % end if occ_flag
    
%     tic;
    
    if frame == 1  %first frame, train with a single image
        model_alphaf = alphaf;
        model_xf = xf;
    else
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
        %           subsequent frames, interpolate model
        if frame == 2
            occ_model_max = occ_max;
            occ_model_APCE = APCE;
            model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
            model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
        else
            if ~occ_flag % occ_flag = false
                occ_flag = APCE < occ_APCE_lambda1 * occ_model_APCE ...
                    && occ_max < occ_max_lambda1 * occ_model_max;
                if ~occ_flag % occ_flag = flase
                    model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
                    model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
                    occ_model_max = occ_update_factor * occ_model_max + (1 - occ_update_factor) * occ_max;
                    occ_model_APCE = occ_update_factor * occ_model_APCE + (1 - occ_update_factor) * APCE;
                else
                    % if occ_flag = true, do not update the model
                    %                         occ_model_rgb = occ_update_factor * occ_model_rgb + (1 - occ_update_factor) * occ_rgb;
                    %                         occ_model_depth = occ_update_factor * occ_model_depth + (1 - occ_update_factor) * occ_depth;
                end
                
            else % occ_flag = true
                re_enter_flag = APCE > occ_APCE_lambda2 * occ_model_APCE ...
                    && occ_max > occ_max_lambda2 * occ_model_max;
                if re_enter_flag % re_enter_flag = true
                    occ_flag = false;
                    %                         model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
                    %                         model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
                    %                         depth_model_alphaf = (1 - interp_factor) * depth_model_alphaf + interp_factor * depth_alphaf;
                    %                         depth_model_xf = (1 - interp_factor) * depth_model_xf + interp_factor * depth_xf;
                    %                         occ_model_rgb = (1 - interp_factor) * occ_model_rgb + interp_factor * occ_rgb;
                    %                         occ_model_depth = (1 - interp_factor) * occ_model_depth + interp_factor * occ_depth;
                else
                    % if the target does not re-enter, just do nothing
                end
                %                     occ_model_rgb = occ_update_factor * occ_model_rgb + (1 - occ_update_factor) * occ_rgb;
                %                     occ_model_depth = occ_update_factor * occ_model_depth + (1 - occ_update_factor) * occ_depth;
            end
            
        end
    end % end of frame == 1
    
%     fprintf('frame %d: occ_rgb = %f, occ_depth = %f\n model_rgb = %f, model_depth = %f, occ_flag = %d' , ...
%         frame, occ_rgb, occ_depth, occ_model_rgb, occ_model_depth, occ_flag);
    
%     depth_time = depth_time + toc() / 2;
    %save position and timing
    time = time + toc();
    positions(frame,:) = pos;
%     total_time = etime(clock,t0);
%     time = time + total_time - depth_time;
    
    if occ_flag
        occ_results(frame) = 1;
    end
    
    box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    rect_results(frame,:)=box;
    %visualization
    if show_visualization
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        
        drawnow
    end
    
end % end of for frame = 1 : length
    
if resize_image
    positions = positions * 2;
    rect_results = rect_results * 2;
end

flag = true;
end

