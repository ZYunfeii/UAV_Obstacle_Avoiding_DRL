% 毕业设计对比算法A* algorithm实现
clc; clear; close all;
%% 参数读取与设置
obstacleMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/obstacleMatrix.csv");
RobstacleMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/RobstacleMatrix.csv")';
% cylinderMatrix = [];
% cylinderRMatrix = [];
% cylinderHMatrix = [];
% coneMatrix = [];
% coneRMatrix = [];
% coneHMatrix = [];
cylinderMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/cylinderMatrix.csv");
cylinderRMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/cylinderRMatrix.csv")';
cylinderHMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/cylinderHMatrix.csv")';
coneMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/coneMatrix.csv");
coneRMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/coneRMatrix.csv");
coneHMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/coneHMatrix.csv");
start = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/start.csv")';
goal = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/goal.csv")';
[numberOfSphere, ~] = size(obstacleMatrix);
[numberOfCylinder, ~] = size(cylinderMatrix);
[numberOfCone,~] = size(coneMatrix);
Alldirec = [[1,0,0];[0,1,0];[0,0,1];[-1,0,0];[0,-1,0];[0,0,-1];...
            [1,1,0];[1,0,1];[0,1,1];[-1,-1,0];[-1,0,-1];[0,-1,-1];...
            [1,-1,0];[-1,1,0];[1,0,-1];[-1,0,1];[0,1,-1];[0,-1,1];...
            [1,1,1];[-1,-1,-1];[1,-1,-1];[-1,1,-1];[-1,-1,1];[1,1,-1];...
            [1,-1,1];[-1,1,1]];
threshold = 0.7;
stop = threshold*1.5;
g = [start, 0; goal, inf]; % 每一行前三个数为点坐标，第四个数为路径耗散
Path = [];
Parent = [];
Open = [start, g(findIndex(g,start),4) + getDist(start,goal)];
%% 绘制障碍环境
figure(1)
for i = 1:numberOfSphere   %绘制静态球障碍物
    drawSphere(obstacleMatrix(i,:), RobstacleMatrix(i))
end

for i = 1:numberOfCylinder   %绘制圆柱体障碍物
    drawCylinder(cylinderMatrix(i,:), cylinderRMatrix(i), cylinderHMatrix(i));
end

for i = 1:numberOfCone       %绘制圆锥障碍物
    drawCone(coneMatrix(i,:), coneRMatrix(i), coneHMatrix(i));
end

bar1 = scatter3(start(1),start(2),start(3),80,"cyan",'filled','o','MarkerEdgeColor','k');hold on
bar2 = scatter3(goal(1),goal(2),goal(3),80,"magenta",'filled',"o",'MarkerEdgeColor','k');
text(start(1),start(2),start(3),'  起点');
text(goal(1),goal(2),goal(3),'  终点');
xlabel('x(m)'); ylabel('y(m)'); zlabel('z(m)');
title('A*算法UAV航迹规划路径');
axis equal
% set(gcf,'unit','centimeters','position',[30 10 20 15]);
%% 主循环
tic;
while ~isempty(Open)
    [xi, index] = findMin(Open);
    Open(index,:) = [];
    if getDist(xi, goal) < stop
        break;
    end
    children = getChildren(xi, Alldirec, threshold, obstacleMatrix, RobstacleMatrix,...
                           cylinderMatrix, cylinderRMatrix, cylinderHMatrix,...
                           coneMatrix, coneRMatrix, coneHMatrix,...
                           numberOfSphere, numberOfCylinder, numberOfCone);
    scatter3(children(:,1),children(:,2),children(:,3),10,'filled','o');
    drawnow;
    [n,~] = size(children);
    for i = 1:n
        child = children(i,:);
        if findIndex(g, child) == 0   % child不在g
            g = [g; child, inf];
        end
        a = g(findIndex(g, xi),4) + getDist(xi,child);
        if a < g(findIndex(g, child),4)
            g(findIndex(g, child),4) = a;
            Parent = setParent(Parent, child,xi);
            Open = setOpen(Open, child, a, goal);
        end
    end  
end
lastPoint = xi;
%% 回溯轨迹
x = lastPoint;
Path = x;
[n,~] = size(Parent);
while any(x ~= start)
    for i = 1:n
        if Parent(i,1:3) == x
            Path = [Parent(i,4:6); Path];
            break;
        end
    end
    x = Parent(i,4:6);
end
bar3 = plot3([Path(:,1);goal(1)],[Path(:,2);goal(2)],[Path(:,3);goal(3)],'LineWidth',3,'color','r');
filPathX = [start(1),MovingAverage(Path(2:end,1),5),goal(1)];
filPathY = [start(2),MovingAverage(Path(2:end,2),5),goal(2)];
filPathZ = [start(3),MovingAverage(Path(2:end,3),5),goal(3)];
bar4 = plot3(filPathX, filPathY, filPathZ,'LineWidth',3,'color','g');
legend([bar1, bar2,bar3,bar4],["起始点","终止点","无人机航迹","MA平滑后航迹"],'Location','northwest');
%% 完整path
path = [Path;goal];
%% 计算轨迹距离
pathLength = 0;
[n,~] = size(Path);
for i = 1:n-1
    pathLength = pathLength + getDist(Path(i,:),Path(i+1,:));
end
pathLength = pathLength + getDist(Path(end,:),goal);
fprintf('计算时间:%f秒\n路径的长度为:%f\n GS:%f°\n LS:%f°',toc,pathLength,calGs(path)/pi*180,calLs(path)/pi*180);
%% 存储轨迹
csvwrite('F:\MasterDegree\毕业设计\实验数据\静态环境轨迹联合绘制/Astar_path.csv',[filPathX',filPathY', filPathZ']);
%% 函数
function children = getChildren(pos, Alldirec, step,circleCenter,circleR,...
                                cylinderCenter,cylinderR, cylinderH,...
                                coneMatrix, coneRMatrix, coneHMatrix,...
                                numberOfSphere, numberOfCylinder, numberOfCone)
allchild = [];
[n,~] = size(Alldirec);
for i = 1:n
    direc = Alldirec(i,:);
    child = pos + direc * step;
    if ~checkCol(child, circleCenter,circleR, cylinderCenter,cylinderR, cylinderH,...
                 coneMatrix, coneRMatrix, coneHMatrix,numberOfSphere, numberOfCylinder, numberOfCone)
        continue;
    end
    allchild = [allchild; child];
end
children = allchild;
end

function flag = checkCol(pos, circleCenter,circleR, cylinderCenter,cylinderR, cylinderH,...
                         coneMatrix, coneRMatrix, coneHMatrix,numberOfSphere, numberOfCylinder, numberOfCone)
flag = true;
for i = 1:numberOfSphere
    if getDist(pos, circleCenter(i,:)) <= circleR(i)
        flag = false;
        break;
    end
end
for i = 1:numberOfCylinder
    if getDist(pos(1:2), cylinderCenter(i,:)) <= cylinderR(i) && pos(3) <= cylinderH(i)
        flag = false;
        break;
    end
end
for i = 1:numberOfCone
    if getDist(pos(1:2), coneMatrix(i,:)) <= coneRMatrix(i) - coneRMatrix(i)/coneHMatrix(i)*pos(3)...
            && pos(3) >= 0
        flag = false;
        break;
    end
end
if pos(3) <= 0, flag = false; end
end

function Par = setParent(Parent, xj, xi)
[n,~] = size(Parent);
if n == 0
    Par = [xj, xi];
else
    for i = 1:n
        if Parent(i,1:3) == xj
            Parent(i,4:6) = xi;
            Par = Parent;
            break;
        end
        if i == n
            Par = [Parent; xj, xi];
        end
    end
end
end

function Ope = setOpen(Open, child, a, goal)
[n,~] = size(Open);
if n == 0
    Ope = [child, a + getDist(child, goal)];
else
    for i = 1:n
        if Open(i,1:3) == child
            Open(i,4) = a + getDist(child, goal);
            Ope = Open;
        end
        if i == n
            Ope = [Open; child, a + getDist(child, goal)];
        end
    end
end
end

function h = heuristic(pos, goal)
h = max([abs(goal(1) - pos(1)),abs(goal(2) - pos(2)),abs(goal(3) - pos(3))]);
end

function index = findIndex(g, pos)
[n,~] = size(g);
index = 0;    % 表示没有找到索引
for i = 1:n
    if g(i,1:3) == pos
        index = i;   % 索引为i
        break;
    end
end
end

function d = getDist(x,y)
d = sqrt(sum((x - y).^2));
end

function [pos, index] = findMin(Open)
[~,index] = min(Open(:,4));
pos = Open(index,1:3);
end



