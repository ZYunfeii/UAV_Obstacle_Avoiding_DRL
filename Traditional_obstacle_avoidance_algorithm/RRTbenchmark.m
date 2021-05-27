% 毕业设计对比算法RRT实现
clc; clear; close all;
%% 参数读取与设置
obstacleMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/obstacleMatrix.csv");
RobstacleMatrix = csvread("../Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/RobstacleMatrix.csv")';
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
stepSize = 0.2;
threshold = 0.2;
maxFailedAttempts = 10000;
searchSize = 1.1*[goal(1) - start(1), goal(2) - start(2), goal(3) - start(3)]; % 这里通过调节前面的因子调节搜索范围
RRTree = double([start, -1]);
failedAttempts = 0;
pathFound = false;
display = true;
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
xlabel('x(km)'); ylabel('y(km)'); zlabel('z(km)');
title('RRT算法UAV航迹规划路径');
axis equal
% set(gcf,'unit','centimeters','position',[30 10 20 15]);
%% 主循环
tic;
while failedAttempts <= maxFailedAttempts
    %% 选择随机点作为延展目标 50%几率随机 50%几率goal
    if rand < 0.5
        sample = rand(1,3).*searchSize + start;
    else
        sample = goal;
    end
    %% 选择RRTree上距离sample最近的一点作为延展根节点
    [A, I] = min( distanceCost(RRTree(:,1:3),sample) ,[],1);
    closestNode = RRTree(I(1),1:3);
    %% 延展RRTree
    movingVec = [sample(1)-closestNode(1),sample(2)-closestNode(2),sample(3)-closestNode(3)];
    movingVec = movingVec/sqrt(sum(movingVec.^2));  %单位化
    newPoint = closestNode + stepSize * movingVec;
    %% 判断延展后的新点newPoint是否满足要求（碰撞检测）
    if ~checkPath(closestNode, newPoint, obstacleMatrix,RobstacleMatrix, cylinderMatrix,...
            cylinderRMatrix, cylinderHMatrix, stepSize, numberOfSphere, numberOfCylinder,...
            numberOfCone, coneMatrix, coneRMatrix, coneHMatrix) 
        failedAttempts = failedAttempts + 1;
        continue;
    end
    % 检测newPoint是否临近目标点
    if distanceCost(newPoint,goal) < threshold, pathFound = true; break; end
    % 如果newPoint与之前RRTree上某一点的距离小于threshold说明newPoint的意义不大，舍弃
    [A, I2] = min( distanceCost(RRTree(:,1:3),newPoint) ,[],1);
    if distanceCost(newPoint,RRTree(I2(1),1:3)) < threshold, failedAttempts = failedAttempts + 1; continue; end
    %% 将newPoint加入RRTree
    RRTree = [RRTree; newPoint I(1)]; % add node
    failedAttempts = 0;
    if display, plot3([closestNode(1);newPoint(1)],[closestNode(2);newPoint(2)],[closestNode(3);newPoint(3)],'LineWidth',2); end
    pause(0.05);
end
if display && pathFound, plot3([closestNode(1);goal(1)],[closestNode(2);goal(2)],[closestNode(3);goal(3)],'LineWidth',2); end
if ~pathFound, error('no path found. maximum attempts reached'); end
%% 回溯轨迹
path = goal;
prev = I(1);
while prev > 0
    path = [RRTree(prev,1:3); path];
    prev = RRTree(prev,4);
end
bar3 = plot3(path(:,1),path(:,2),path(:,3),'LineWidth',3,'color','r');
filPathX = [start(1),MovingAverage(path(2:end-1,1),5),goal(1)];
filPathY = [start(2),MovingAverage(path(2:end-1,2),5),goal(2)];
filPathZ = [start(3),MovingAverage(path(2:end-1,3),5),goal(3)];
bar4 = plot3(filPathX,filPathY,filPathZ,'LineWidth',3,'color','g');
legend([bar1,bar2,bar3,bar4],["起始点","终止点","无人机航迹","MA平滑后航迹"],'Location','northwest');
%% 存储轨迹
csvwrite('F:\MasterDegree\毕业设计\实验数据\静态环境轨迹联合绘制/rrt_path.csv',[filPathX',filPathY', filPathZ']);
%% 计算轨迹长度以及解算时间
pathLength = 0;
for i=1:length(path(:,1))-1, pathLength = pathLength + distanceCost(path(i,1:3),path(i+1,1:3)); end % calculate path length
fprintf('运行时间：%d \n路径长度=%d\n GS:%f°\n LS:%f°', toc, pathLength,calGs(path)/pi*180,calLs(path)/pi*180);
%% 碰撞检测函数
function flag = checkPath(n, newPos, circleCenter,circleR, cylinderCenter,...
    cylinderR, cylinderH, step, numberOfSphere, numberOfCylinder, numberOfCone,...
    coneMatrix, coneRMatrix, coneHMatrix)
flag = true;
movingVec = [newPos(1)-n(1),newPos(2)-n(2),newPos(3)-n(3)];
movingVec = movingVec/sqrt(sum(movingVec.^2)); %单位化
for R = 0:step/5:distanceCost(n, newPos)
    posCheck = n + R.*movingVec;
    for i = 1:numberOfSphere
        if distanceCost(posCheck, circleCenter(i,:)) <= circleR(i)
            flag = false;
            break;
        end
    end
    for i = 1:numberOfCylinder
        if distanceCost(posCheck(1:2), cylinderCenter(i,:)) <= cylinderR(i) && posCheck(3) <= cylinderH(i)...
           && posCheck(3) >= 0
            flag = false;
            break;
        end
    end
    for i = 1:numberOfCone
        if distanceCost(posCheck(1:2), coneMatrix(i,:)) <= coneRMatrix(i) - coneRMatrix(i)/coneHMatrix(i)*posCheck(3)...
           && posCheck(3) >= 0
            flag = false;
            break;
        end
    end
end
end

function h=distanceCost(a,b)     
	h = sqrt(sum((a-b).^2, 2));
end

















