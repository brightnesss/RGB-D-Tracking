function [ imgs ] = rgbdimgread( input_str )
%RGBIMGREAD read rgb and depth images from given address
%   RGBIMGREAD loads all the relevant information for the
%   sequence depth and video data in the given path for the Princeton RGB-D
%   dataset.Note that depth images are 16-bit and need extrea processing.

frames_str = [input_str,'\frames.mat'];

frames = load(frames_str);

frames = frames.frames;

length = frames.length;

imgs.rgb = cell(length,1);
imgs.depth = cell(length,1);

for frameId = 1 : length
   rgbimageName = fullfile(input_str,sprintf('rgb/r-%d-%d.png', frames.imageTimestamp(frameId), frames.imageFrameID(frameId)));  
   rgbimg = imread(rgbimageName);  
   imgs.rgb{frameId} = rgbimg;
   depthimgName = fullfile(input_str,sprintf('depth/d-%d-%d.png', frames.depthTimestamp(frameId), frames.depthFrameID(frameId)));
   depthimg = imread(depthimgName);
   depthimg = bitor(bitshift(depthimg,-3), bitshift(depthimg,16-3)); 
   
   %Normalize depth data as a grayscale image [0 255]
   depthimg = double(depthimg);  
   depthimg(depthimg==0) = 10000;
   depthimg = (depthimg-500)/8500;%only use the data from 0.5-8m
   depthimg(depthimg<0) = 0;
   depthimg(depthimg>1) = 1;
   depthimg = uint8(255*(1 - depthimg));
   imgs.depth{frameId} = depthimg;
end


end

