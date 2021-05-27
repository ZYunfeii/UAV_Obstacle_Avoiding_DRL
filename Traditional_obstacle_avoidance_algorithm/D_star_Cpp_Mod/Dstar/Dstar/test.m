clc;clear;close all;
uav_path = csvread("path.csv");
obs_trace_path = csvread("obs_trace_path.csv");
obs_trace_path = obs_trace_path(1:size(uav_path,1),:);
for i = 1:size(uav_path,1)-1
    plot3([uav_path(i,1),uav_path(i+1,1)],[uav_path(i,2),uav_path(i+1,2)],[uav_path(i,3),uav_path(i+1,3)],'LineWidth',2,"Color",'r');
    axis equal;
    grid on;
    hold on;
    try delete(h) 
    catch 
    end
    h = drawSphere(obs_trace_path(i,:),7.5);
    pause(0.1);
end


function h = drawSphere(pos, r)
[x,y,z] = sphere(60);
h = surfc(r*x+pos(1), r*y+pos(2), r*z+pos(3));
hold on;
end