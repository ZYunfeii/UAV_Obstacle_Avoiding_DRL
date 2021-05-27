clc;clear;close all;
%% 读取数据
obs_trace = csvread("obs_trace.csv");
path = csvread("path.csv");
obs_r = csvread("obs_r.csv");
start = path(1,:);
goal = path(end,:);
%% 绘制
bar1 = scatter3(start(1),start(2),start(3),80,"cyan",'filled','o');hold on
bar2 = scatter3(goal(1),goal(2),goal(3),80,"magenta",'filled',"o");
axis equal;
grid on;
%% 循环
for i = 1:size(path,1)-1
    try delete(sbar) 
    catch
    end
    sbar = drawSphere(obs_trace(i,:),obs_r);
    plot3([path(i,1),path(i+1,1)],[path(i,2),path(i+1,2)],[path(i,3),path(i+1,3)],"LineWidth",2,"Color","r");
    pause(0.1);
end

%% 函数
% 球绘制函数
function h = drawSphere(pos, r)
[x,y,z] = sphere(60);
h = surfc(r*x+pos(1), r*y+pos(2), r*z+pos(3));
hold on;
end