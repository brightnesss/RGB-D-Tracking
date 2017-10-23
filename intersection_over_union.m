function [ scalar ] = intersection_over_union( propose, groundtruth )
%UNTITLED calcuate the intersection-over-union(IoU) with given rectangle
%position
%   Input: propose is the results from your algorithms and groundtruth is the
%   groundtruth. They can be both specify one rectangle or matrices where
%   each row is a position vector. And the num of rectangles of propose and
%   groundtruth must be equal.
%   Output: if inputs are just one rectangle's position, Area is a scalar.
%   or if inputs are matrices, then Area is a vector and Area(i) is the
%   intersection-over-union scalar of i-th rectangle.

if size(propose,1)~=size(groundtruth,1)
    if size(groundtruth,1)<size(propose,1)
        propose=propose(1:size(groundtruth,1),:);
    end
end


rectintarea = rectint(propose, groundtruth);

rectintarea = diag(rectintarea);

propose_area = propose(:,3) .* propose(:,4);

groundtruth_area = groundtruth(:,3) .* groundtruth(:,4);

union_area = propose_area + groundtruth_area - rectintarea;

scalar = rectintarea ./ union_area;

scalar(find(union_area == 0)) = 0;

end

