function [positions, rect_results, time] = tracker(video_path, rgbdimgs, pos, target_sz, ...
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
	if show_visualization  %create video interface
		update_visualization = show_video(rgbdimgs.rgb, video_path, resize_image);
	end
	
	
	%note: variables ending with 'f' are in the Fourier domain.
    length = numel(rgbdimgs.rgb);
	time = 0;  %to calculate FPS
	positions = zeros(length, 2);  %to calculate precision
	rect_results = zeros(length, 4);  %to calculate 
    response = zeros(size(cos_window,1),size(cos_window,2),size(search_size,2));
    szid = 1;
    param1 = zeros(size(search_size,2), 6);
    occ_flag = false;   % whether the target is occluded
    occ_rgb_lambda1 = 0.4;  % occ_flag from false to true
    occ_rgb_lambda2 = 0.6;  % occ_flag from true to false  
    occ_depth_lambda1 = 0.4;  % occ_flag from false to true
    occ_depth_lambda2 = 0.6;  % occ_flag from true to false
    
%     p = gcp('nocreate');   
%     if isempty(p)
%         parpool('local');
%     end
    
	for frame = 1 : length
		%load image
		rgbim = rgbdimgs.rgb{frame};
        depthim = rgbdimgs.depth{frame};
        if resize_image
            rgbim = imresize(rgbim, 0.5);
            depthim = imresize(depthim, 0.5);
        end
           
        t0 = clock;
        depth_time = 0;

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
                patch = uint8(warpimg(double(rgbim), param1(i,:), window_sz));
                zf = fft2(get_features(patch, features, cell_size, cos_window,w2c));

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
            end
			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[~,tmp, ~] = find(response == max(response(:)), 1);

            szid = floor((tmp-1)/(size(cos_window,2)))+1;
            
            rgbmaxresponse = response(:,:,szid);
            
            %再在depth通道上进行处理
            %obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is given by rgb-image)
%             depth_patch = get_subwindow(depthim, pos, target_sz * search_size(szid));
%             tmp_sz = floor((target_sz * (1 + padding)) * search_size(szid));
%             param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
%                         tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
%             param0 = affparam2mat(param0);
            tic;
            depth_patch = uint8(warpimg(double(depthim), param1(szid,:), window_sz));
            depth_zf = fft2(get_features(depth_patch, 'hog', cell_size, cos_window));
            %calculate response of the classifier at all shifts
			switch kernel.type
			case 'gaussian'
				depth_kzf = gaussian_correlation(depth_zf, depth_model_xf, kernel.sigma);
			case 'polynomial'
				depth_kzf = polynomial_correlation(depth_zf, depth_model_xf, kernel.poly_a, kernel.poly_b);
			case 'linear'
				depth_kzf = linear_correlation(depth_zf, depth_model_xf);
			end
			depth_response = real(ifft2(depth_model_alphaf .* depth_kzf));  %equation for fast detection
            depth_time = depth_time + toc();
            
            % calculate occlusion paramter
            %max response of rgb respnse map
            occ_rgb = max(rgbmaxresponse(:));
            
            %average peak-to-correlation energy(APCE)
            occ_depth = (max(depth_response(:)) - min(depth_response(:))) ^ 2 ...
                / mean(mean((depth_response - min(depth_response(:))) .^ 2)); 
            
            %两种计算结果进行融合
            final_response = rgbmaxresponse + depth_response;
            
            % for test the response map
%             f2 = figure(2);
%             imshow(fftshift(response(:,:,szid)));
%             set(f2,'position',[500,100,600,600]);
%             f3 = figure(3);
%             imshow(fftshift(depth_response));
%             set(f3,'position',[500,100,600,600]);
%             f4 = figure(4);
%             imshow(fftshift(final_response));
%             set(f4,'position',[500,100,600,600]);
%             
%             if frame >= 36
%                 pause;
%             end
            
            %target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[vert_delta, horiz_delta] = find(final_response == max(final_response(:)), 1);
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
		end

		%obtain a subwindow for training at newly estimated target position
        
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
        patch = uint8(warpimg(double(rgbim), param0, window_sz));
        %为了统一hog和cn的维度，以hog的维度为准，将patch降维到hog特征维度
        x = get_features(patch, features, cell_size, cos_window,w2c);
        xf = fft2(x);
        
        % 处理深度图,采用KCF那一套写法,默认采用hog特征
        %         depth_patch = get_subwindow(depthim, pos, target_sz);
        tic;
        depth_patch = uint8(warpimg(double(depthim), param0, window_sz));
        depth_xf = fft2(get_features(depth_patch, 'hog', cell_size, cos_window, 0));
        
        %Kernel Ridge Regression, calculate alphas (in Fourier domain)
        switch kernel.type
            case 'gaussian'
                kf = gaussian_correlation(xf, xf, kernel.sigma);
                depth_kf = gaussian_correlation(depth_xf, depth_xf, kernel.sigma);
            case 'polynomial'
                kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
                depth_kf = polynomial_correlation(depth_xf, depth_xf, kernel.poly_a, kernel.poly_b);
            case 'linear'
                kf = linear_correlation(xf, xf);
                depth_kf = linear_correlation(depth_xf, depth_xf);
        end
        alphaf = yf ./ (kf + lambda);   %equation for fast training
        depth_alphaf = yf ./ (depth_kf + lambda);
        
        if frame == 1  %first frame, train with a single image
            model_alphaf = alphaf;
            model_xf = xf;
            depth_model_alphaf = depth_alphaf;
            depth_model_xf = depth_xf;
        else
            model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
            model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
            depth_model_alphaf = (1 - interp_factor) * depth_model_alphaf + interp_factor * depth_alphaf;
            depth_model_xf = (1 - interp_factor) * depth_model_xf + interp_factor * depth_xf;
            %           subsequent frames, interpolate model
            if frame == 2
                occ_model_rgb = occ_rgb;
                occ_model_depth = occ_depth;
                model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
                model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
                depth_model_alphaf = (1 - interp_factor) * depth_model_alphaf + interp_factor * depth_alphaf;
                depth_model_xf = (1 - interp_factor) * depth_model_xf + interp_factor * depth_xf;
            else
                if ~occ_flag % occ_flag = false
                    occ_flag = occ_depth < occ_depth_lambda1 * occ_model_depth ...
                        && occ_rgb < occ_rgb_lambda1 * occ_model_rgb;
                    if ~occ_flag % occ_flag = flase
                        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
                        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
                        depth_model_alphaf = (1 - interp_factor) * depth_model_alphaf + interp_factor * depth_alphaf;
                        depth_model_xf = (1 - interp_factor) * depth_model_xf + interp_factor * depth_xf;
                        occ_model_rgb = (1 - interp_factor) * occ_model_rgb + interp_factor * occ_rgb;
                        occ_model_depth = (1 - interp_factor) * occ_model_depth + interp_factor * occ_depth;
                    else
                        % if occ_flag = true, do not update the model
                    end
                else % occ_flag = true
                    re_enter_flag = occ_depth > occ_depth_lambda2 * occ_model_depth ...
                        && occ_rgb > occ_rgb_lambda2 * occ_model_rgb;
                    if re_enter_flag % re_enter_flag = true
                        occ_flag = false;
                        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
                        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
                        depth_model_alphaf = (1 - interp_factor) * depth_model_alphaf + interp_factor * depth_alphaf;
                        depth_model_xf = (1 - interp_factor) * depth_model_xf + interp_factor * depth_xf;
                        occ_model_rgb = (1 - interp_factor) * occ_model_rgb + interp_factor * occ_rgb;
                        occ_model_depth = (1 - interp_factor) * occ_model_depth + interp_factor * occ_depth;
                    else
                        % if the target does not re-enter, just do nothing
                    end
                end
                
            end
        end
        depth_time = depth_time + toc() / 2;
        %save position and timing
        positions(frame,:) = pos;
        total_time = etime(clock,t0);
        time = time + total_time - depth_time;
        
        
        
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        rect_results(frame,:)=box;
        %visualization
        if show_visualization
            stop = update_visualization(frame, box);
            if stop, break, end  %user pressed Esc, stop early
            
            drawnow
        end
        
    end
    
    if resize_image
        positions = positions * 2;
        rect_results = rect_results*2;
    end
end

