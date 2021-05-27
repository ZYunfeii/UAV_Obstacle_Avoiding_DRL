% IIFDS算法实现
clc; clear; close all;
%% 参考文献：Formation Obstacle Avoidance: A Fluid-Based Solution
%% 初始参数设置
V0 = 1; 
threshold = 0.2;
stepSize = 0.1;
%% demo1:静态单球形障碍
% start = [0,1,5];
% goal = [10,10,6];
% obsCenter = [5,5,5.5];
% obsR = 1;
% row0 = 0.5;   % 反应系数
% sigma0 = 0.5; % 切向反应系数
% theta = 0;    % 切向方向系数
% % 绘制环境
% scatter3(start(1),start(2),start(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
% scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')
% text(start(1),start(2),start(3),'  起点');
% text(goal(1),goal(2),goal(3),'  终点');
% drawSphere(obsCenter,obsR);
% xlabel('x(m)'); ylabel('y(m)'); zlabel('z(m)');
% title('UAV航迹规划路径'); axis equal;
% % 开始
% pos = start;
% path = pos;
% bar = [];
% for kk = 1:6
%     row0 = row0 + 0.2;
%     sigma0 = sigma0 + 0.2;
%     theta = theta + 0.2;
%     pos = start;
%     path = pos;
%     while distanceCost(pos, goal) > threshold
%         u = initField(pos, V0, goal);
%         % 计算排斥矩阵
%         n = partialDerivativeSphere(obsCenter, pos, obsR);
%         tempD = distanceCost(pos, obsCenter) - obsR; % 到障碍物表面距离
%         row = row0 * exp(1-1/(distanceCost(pos, goal)*tempD));
%         T = calculateT(obsCenter, pos, obsR);
%         repulsiveMatrix = - n * n' / T^(1/row) / (n'*n);
%         %计算切向矩阵
%         partialX = (pos(1) - obsCenter(1))*2/obsR^2;
%         partialY = (pos(2) - obsCenter(2))*2/obsR^2;
%         partialZ = (pos(3) - obsCenter(3))*2/obsR^2;
%         tk1 = [partialY, -partialX, 0]';
%         tk2 = [partialX*partialZ, partialY*partialZ, -partialX^2-partialY^2]';
%         tk = trans([cos(theta), sin(theta), 0], tk1, tk2, n);
%         sigma = sigma0 * exp(1-1/(distanceCost(pos, goal)*tempD));
%         tangentialMatrix = tk * n' / T^(1/sigma) / calVecLen(tk) / calVecLen(n);
%         % 计算修正后扰动矩阵
%         M = eye(3) + repulsiveMatrix + tangentialMatrix;
%         ubar = (M * u)';
%         nextPos = pos + ubar * stepSize;
%         path = [path;nextPos];
%         pos = nextPos;
%     end
%     path = [path;goal];
%     b = plot3(path(:,1), path(:,2), path(:,3),'LineWidth',2); hold on;
%     bar = [bar,b];
% end
% legend(bar,["=","=","=","=","=","="]);
%% demo2:单个动态球障碍
start = [1,4,5];
goal = [10,10,5.5];
obsStart = [5,5,5]; % 动态障碍物开始的点位
obsR = 1.5;           % 动态障碍物的半径
lambda = 8;
row0 = 0.5;   % 反应系数
sigma0 = 0.5; % 切向反应系数
theta = 0.5;    % 切向方向系数
scatter3(start(1),start(2),start(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')
text(start(1),start(2),start(3),'  起点');
text(goal(1),goal(2),goal(3),'  终点');
xlabel('x(km)'); ylabel('y(km)'); zlabel('z(km)');
title('UAV动态航迹规划路径'); axis equal;
uavPos = start;
path = uavPos;
t = 0; % 初始化时间
timeStep = 0.1;
flag = true; % 是否通过碰撞检测
collPoint = [];
while distanceCost(uavPos, goal) > threshold
    obsCenter = [obsStart(1)+2*cos(t),obsStart(2)+2*sin(t),obsStart(3)];
    vObs = [-2*sin(t),2*cos(t),0]; % vObs是obsPos对t的导数
    try delete(B1), catch, end
    try delete(B2), catch, end
    B1 = drawSphere(obsCenter, obsR);
    B2 = scatter3(uavPos(1),uavPos(2),uavPos(3),80,'filled',"^",'MarkerFaceColor','g'...
                  ,'MarkerEdgeColor','k');
    b1 = plot3([obsStart(1)+2*cos(t-timeStep),obsCenter(1)],[obsStart(2)+2*sin(t-timeStep),obsCenter(2)]...
              ,[obsStart(3),obsStart(3)],'LineWidth',2,'color','b');
    drawnow;
    % 碰撞检测
    if distanceCost(uavPos, obsCenter) <= obsR
        flag = false;
        collPoint = [collPoint; uavPos];
    end
    % 算法开始
    u = initField(uavPos,V0,goal);
    % 计算排斥矩阵
    n = partialDerivativeSphere(obsCenter, uavPos, obsR);
    tempD = distanceCost(uavPos, obsCenter) - obsR; % 到障碍物表面距离
    row = row0 * exp(1-1/(distanceCost(uavPos, goal)*tempD));
    T = calculateT(obsCenter, uavPos, obsR);
    repulsiveMatrix = - n * n' / T^(1/row) / (n'*n);
    %计算切向矩阵
    partialX = (uavPos(1) - obsCenter(1))*2/obsR^2;
    partialY = (uavPos(2) - obsCenter(2))*2/obsR^2;
    partialZ = (uavPos(3) - obsCenter(3))*2/obsR^2;
    tk1 = [partialY, -partialX, 0]';
    tk2 = [partialX*partialZ, partialY*partialZ, -partialX^2-partialY^2]';
    tk = trans([cos(theta), sin(theta), 0], tk1, tk2, n);
    sigma = sigma0 * exp(1-1/(distanceCost(uavPos, goal)*tempD));
    tangentialMatrix = tk * n' / T^(1/sigma) / calVecLen(tk) / calVecLen(n);
    % 计算vp
    vp = exp(-T/lambda)*vObs; 
    % 计算修正后扰动矩阵
    M = eye(3) + repulsiveMatrix + tangentialMatrix;
    ubar = (M * (u-vp'))'+vp;
    uavNextPos = uavPos + ubar * stepSize;
    b2 = plot3([uavPos(1),uavNextPos(1)],[uavPos(2),uavNextPos(2)],[uavPos(3),uavNextPos(3)],'LineWidth',2,'color','r');
    path = [path;uavNextPos];
    uavPos = uavNextPos;
    t = t + timeStep; 
    pause(0.01)
    if t == 0 + timeStep
        legend([b1,b2,B2],["障碍物移动轨迹","UAV规划航路","UAV"],'AutoUpdate','off','Location','west')
    end
end
path = [path;goal];
plot3([path(end,1),goal(1)],[path(end,2),goal(2)],[path(end,3),goal(3)],'LineWidth',2,'color','r');
if flag == true
    fprintf('通过碰撞检测！')
else
    fprintf('未通过碰撞检测！\n')
    disp(collPoint);
end
%% 计算GS,LS,L
pathLength = 0;
for i=1:length(path(:,1))-1, pathLength = pathLength + distanceCost(path(i,1:3),path(i+1,1:3)); end
fprintf("航路长度为:%f\n GS:%f °\n LS:%f °",pathLength, calGs(path)/pi*180, calLs(path)/pi*180);
%% 函数
% 欧式距离求解函数
function h=distanceCost(a,b)
h = sqrt(sum((a-b).^2, 2));
end
% 初始流场求解函数
function u = initField(pos, C, goal)
u = -[(pos(1)-goal(1)), (pos(2)-goal(2)), (pos(3)-goal(3))]' * C / distanceCost(pos, goal);
end
% 球绘制函数
function bar = drawSphere(pos, r)
[x,y,z] = sphere(60);
bar = surf(r*x+pos(1), r*y+pos(2), r*z+pos(3));
hold on;
end
% 球障碍物方程偏导计算函数
function pd = partialDerivativeSphere(obs, pos, r)
pd = [pos(1)-obs(1), pos(2)-obs(2), pos(3)-obs(3)]'*2/r^2;
end
% 球障碍物计算T值
function T = calculateT(obs, pos, r)
T = ((pos(1)-obs(1))^2+(pos(2)-obs(2))^2+(pos(3)-obs(3))^2)/r^2;
end
% 坐标变换后地球坐标下坐标
% newX, newY, newZ是新坐标下三个轴上的方向向量
function pos = trans(originalPoint, xNew, yNew, zNew)
cosa1 = xNew(1)/sqrt(sum(xNew.^2));
cosb1 = xNew(2)/sqrt(sum(xNew.^2));
cosc1 = xNew(3)/sqrt(sum(xNew.^2));

cosa2 = yNew(1)/sqrt(sum(yNew.^2));
cosb2 = yNew(2)/sqrt(sum(yNew.^2));
cosc2 = yNew(3)/sqrt(sum(yNew.^2));

cosa3 = zNew(1)/sqrt(sum(zNew.^2));
cosb3 = zNew(2)/sqrt(sum(zNew.^2));
cosc3 = zNew(3)/sqrt(sum(zNew.^2));

B = [cosa1, cosb1, cosc1;
    cosa2, cosb2, cosc2;
    cosa3, cosb3, cosc3];
pos = B\originalPoint';
end
% 求向量模长
function L = calVecLen(vec)
L = sqrt(sum(vec.^2));
end






