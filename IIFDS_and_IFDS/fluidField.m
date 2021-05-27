% 毕业设计流场避障demo
clc; clear; close all;
%% 初始参数设置
C = 1; 
row = 1.2;
threshold = 0.2; 
%% demo1:无动态障碍，多轨迹绘制，演示基础算法作用
start = [10,10,10.5];
goal = [0,0,10];
obstacleMatrix = [2,5,10;6,2,10;8,5,10.5];
obstacleRMatrix = [1,1.3,1.2];
[numberOfSphere,~] = size(obstacleMatrix);
% 绘制障碍球
% for i = 1:numberOfSphere
%     drawSphere(obstacleMatrix(i,:),obstacleRMatrix(i));
% end
% xlabel('x(m)'); ylabel('y(m)'); zlabel('z(m)');
% scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')

% 测试点收集
testPoint = [];
for x = goal(1):start(1)
    for y = start(2)
        testPoint = [testPoint; x,y,10.5];
    end
end
for y = goal(2):start(2)
    for x = start(1)
        testPoint = [testPoint; x,y,10.5];
    end
end
% 
% % 初始流场计算
% [n,~] = size(testPoint);
% for i = 1:n
%     Pos = testPoint(i,:);
%     path = Pos;
%     while distanceCost(Pos, goal) > threshold
%         u = initField(Pos, C, goal)';
%         nextPos = Pos + u*0.1;
%         path = [path; nextPos];
%         Pos = nextPos;
%     end
%     path = [path; goal];
%     plot3(path(:,1), path(:,2), path(:,3),'LineWidth',2,'color','b'); hold on;
% %     quiver3(path(1:2:end,1),path(1:2:end,2),path(1:2:end,3),...
% %             gradient(path(1:2:end,1)),gradient(path(1:2:end,2)),gradient(path(1:2:end,3)),...
% %             'linewidth',0.5); hold on;
% end
scatter3(goal(1),goal(2),goal(3),120,"magenta",'filled',"o",'MarkerEdgeColor','k')
text(goal(1),goal(2),goal(3),'  终点');
grid on; axis equal;

% 计算扰动场
[n,~] = size(testPoint);
for i = 1:n
    Pos = testPoint(i,:);
    path = Pos;
    while distanceCost(Pos, goal) > threshold
        u = initField(Pos, C, goal); % 初始流场计算
        M = eye(3);
        for j = 1:numberOfSphere
            nw = partialDerivativeSphere(obstacleMatrix(j,:),Pos,obstacleRMatrix(j));
            Tw = calculateT(obstacleMatrix(j,:),Pos,obstacleRMatrix(j));
            if numberOfSphere == 1, omega = 1; 
            else
                omega = 1;
                for k = 1:numberOfSphere
                    if k == j, continue; 
                    else
                        temp = calculateT(obstacleMatrix(k,:),Pos,obstacleRMatrix(k))-1;
                        omega = omega * (temp-1)/((temp-1)+(Tw-1));                              
                    end
                end
            end
            Mmul = eye(3) - omega * nw * nw' / Tw^(1/row) / (nw' * nw);
            M = M * Mmul;
        end
        uc = (M*u)';
        nextPos = Pos + uc * 0.1;
        path = [path; nextPos];
        Pos = nextPos;
    end
    path = [path; goal];
    plot3(path(:,1), path(:,2), path(:,3),'LineWidth',2,'color','b'); hold on;
end

%% demo2:动态障碍(单球形障碍)，单轨迹绘制（从start点开始）
% start = [0,2,5];
% goal = [10,10,5.5];
% obsStart = [5,5,5]; % 动态障碍物开始的点位
% obsR = 1.5;           % 动态障碍物的半径
% lambda = 10;
% scatter3(start(1),start(2),start(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
% scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')
% text(start(1),start(2),start(3),'  起点');
% text(goal(1),goal(2),goal(3),'  终点');
% xlabel('x(m)'); ylabel('y(m)'); zlabel('z(m)');
% title('UAV动态航迹规划路径'); axis equal;
% uavPos = start;
% path = uavPos;
% t = 0; % 初始化时间
% timeStep = 0.1;
% while distanceCost(uavPos, goal) > threshold
%     t = t + timeStep; 
%     obsPos = [obsStart(1)+2*cos(t),obsStart(2)+2*sin(t),obsStart(3)];
%     vObs = [-2*sin(t),2*cos(t),0]; % vObs是obsPos对t的导数
%     try delete(B), catch, end
%     B = drawSphere(obsPos, obsR);
%     drawnow;
%     % 算法开始
% %     u = (distanceCost(uavPos,goal)/distanceCost(start,goal))^1.0 * initField(uavPos,C,goal);
%     u = initField(uavPos,C,goal);
%     Tw = calculateT(obsPos, uavPos, obsR);
%     vp = exp(-Tw/lambda)*vObs;   
%     nw = partialDerivativeSphere(obsPos,uavPos,obsR);
%     M = eye(3) - nw * nw' / Tw^(1/row) / (nw' * nw);
%     ubar = (M * (u-vp'))' + vp;
%     uavNextPos = uavPos + ubar * timeStep;
%     plot3([uavPos(1),uavNextPos(1)],[uavPos(2),uavNextPos(2)],[uavPos(3),uavNextPos(3)],'LineWidth',2,'color','r');
%     path = [path; uavNextPos];
%     uavPos = uavNextPos;
%     pause(0.01)
% end
% plot3([path(end,1),goal(1)],[path(end,2),goal(2)],[path(end,3),goal(3)],'LineWidth',2,'color','r');
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
bar = surfc(r*x+pos(1), r*y+pos(2), r*z+pos(3));
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






