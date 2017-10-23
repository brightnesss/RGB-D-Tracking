function [ success, auc ] = success_plot( track_result, ground_truth, title, show )
%SUCCESS_PLOT 
%   Calculates success for a series of intersection-over-union(IoU)
%   thresholds (percentage of frames where the IoU is larger than a certain
%   threshold).
%   The results are shown in a new figure if SHOW is true.
%
%   Accepts positions and ground truth as Nx2 matrices (for N frames), and
%   a title string.

scalar = intersection_over_union(track_result, ground_truth); 

scalar = scalar(~isnan(scalar));

max_threshold = 1;

success = zeros(51,1);

for i = 0 : 50
    success(i+1) = sum(scalar>=(0.02*i)) / numel(scalar);
end

success = success * 100;

%plot the success
if show == 1
    figure('Name',['Success - ' title])
    plot([0:0.02:1],success, 'k-', 'LineWidth',2)
    xlabel('IoU threshold'), ylabel('Success rate')
end

auc = sum(0.02 * success);

end

