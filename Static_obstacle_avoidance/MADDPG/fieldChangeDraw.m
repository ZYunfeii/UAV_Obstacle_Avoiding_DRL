clc; clear; close all;
%% 人工势场参数
dgoal = 5; r0 = 5;  
epsilon = 1; % 引力因子
eta = 0.5;
obstacle = [12,12,12];
obstacleMatrix = csvread("./data_csv/obstacleMatrix.csv");
cylinderMatrix = csvread("./data_csv/cylinderMatrix.csv");
qgoal = csvread("./data_csv/goal.csv")';
start = csvread("./data_csv/start.csv");
actionCurve1List = csvread("./data_csv/actionCurve1List.csv");
actionCurve2List = csvread("./data_csv/actionCurve2List.csv");
[step,~] = size(actionCurve1List);
%% 引力场的绘制
% 由于所使用的APF算法是三维的算法，无法绘制四维的势场，拟采用分别绘制三个三维势场表示。
%xoy面分解引力和斥力
interval = 0.1;
[x, y] = meshgrid(0:interval:10, 0:interval:8);
z1 = attractionField(x,y,epsilon,dgoal,qgoal);
[n1,~] = size(obstacleMatrix); 
[u, v] = size(x);
figure(1);
for time = 1:step
    z2 = zeros(u,v);
    for i = 1:n1
        z2 = z2 + repulsionField(x,y,actionCurve1List(time,i),r0,obstacleMatrix(i,:),qgoal);
    end
    z = z1 + z2;
    try delete(bar) 
    catch
    end
    bar = mesh(x,y,z);title('引力场+斥力场');
    view([20,40]);
    drawnow;
    pause(0.2)
end
title('引力场+斥力场');hold on;
index1 = round(qgoal(1)/interval)+1; index2 = round(qgoal(2)/interval)+1; 
h1 = scatter3(qgoal(1),qgoal(2),z(index2,index1),180,"cyan",'filled','o');
index1 = round(start(1)/interval)+1; index2 = round(start(2)/interval)+1; 
h2 = scatter3(start(1),start(2),z(index2,index1),180,"red",'filled','o');
legend([h1,h2],["目标点","起始点"])




%% 函数
function Att = attractionField(x,y,epsilon,dgoal,qgoal)
[n, m] = size(x);
Att = [];
for i = 1:n
    for j = 1:m
        position = [x(i,j), y(i,j)];
        r = sqrt(sum((position - qgoal(1:2)).^2));
        if r <= dgoal
            Att(i,j) = 0.5 * epsilon * r^2;
        else
            Att(i,j) = dgoal * epsilon * r - 0.5 * epsilon * dgoal^2;
        end
    end
end
end

function Rep = repulsionField(x,y,eta,r0,obstacle,qgoal)
[n,m] = size(x);
thre = 0.5;
Rep = [];
for i = 1:n
    for j = 1:m
        position = [x(i,j), y(i,j)];
        r = sqrt(sum((position - obstacle(1:2)).^2));
        r_ = sqrt(sum((position - qgoal(1:2)).^2));
        if (r <= r0) && (r >= thre)
            Rep(i,j) = 0.5 * eta * (1/r - 1/r0)^2 * r_^2;
        elseif r < thre
            Rep(i,j) = 0.5 * eta * (1/thre - 1/r0)^2 * r_^2;
        else
            Rep(i,j) = 0;
        end
    end
end
end













