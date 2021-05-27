% 毕业设计对比算法蚁群算法实现
clc; clear; close all;
rand('seed',5);
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

popNumber = 50;  % 蚁群个体数量
rou = 0.1;        % 挥发因子
bestFitness = []; % 每一代的最佳适应值储存列表
bestfitness = inf;% 初始化最佳适应值（本案例中越小越好）
everyIterFitness = [];
deltaX = 0.2; deltaY = 0.2; deltaZ = 0.2;
gridXNumber = floor(abs(goal(1) - start(1)) / deltaX);
gridYNumber = 80; gridZNumber = 80;
ybegin = start(2) - 20*deltaY; zbegin = start(3) - 20*deltaZ;
pheromone = ones(gridXNumber, gridYNumber, gridZNumber);
ycMax = 3; zcMax = 3; % 蚂蚁沿y轴最大变动格子数和沿z轴最大变动格子数
bestPath = []; 
iterMax = 80; 
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
title('蚁群算法UAV航迹规划路径');
axis equal
% set(gcf,'unit','centimeters','position',[30 10 20 15]);
%% 主循环
tic;
for iter = 1:iterMax
    fprintf("程序已运行：%.2f%%\n",iter/iterMax*100);
    % 路径搜索
    [path, pheromone] = searchPath(popNumber, pheromone, start, goal, ycMax, zcMax,...
                                   deltaX, deltaY, deltaZ, obstacleMatrix,RobstacleMatrix,...
                                   cylinderMatrix, cylinderRMatrix, cylinderHMatrix,...
                                   coneMatrix, coneRMatrix, coneHMatrix,...
                                   ybegin, zbegin, gridYNumber, gridZNumber);
    % 路径适应值计算
    fitness = calFit(path, deltaX, start, goal);
    [newBestFitness, bestIndex] = min(fitness);
    everyIterFitness = [everyIterFitness, newBestFitness];
    if newBestFitness < bestfitness
        bestfitness = newBestFitness;
        bestPath = path(bestIndex, :, :);
    end
    bestFitness = [bestFitness, bestfitness];
    % 更新信息素
    cfit = 100 / bestfitness;
    iterNum = 0;
    for x = start(1) + deltaX : deltaX : goal(1) - 0.001
        iterNum = iterNum + 1;
        pheromone(iterNum, round((bestPath(:,iterNum+1,1)-ybegin)/deltaY), round((bestPath(:,iterNum+1,2)-zbegin)/deltaZ))...
        = (1 - rou) * pheromone(iterNum, round((bestPath(:,iterNum+1,1)-ybegin)/deltaY), round((bestPath(:,iterNum+1,2)-zbegin)/deltaZ)) + cfit;
    end
%     for ant = 1:popNumber %对每一条蚂蚁的路径
%         cfit = fitness(ant);
%         iterNum = 0;
%         for x = start(1) + deltaX : deltaX : goal(1) - 0.001
%             iterNum = iterNum + 1;
%             pheromone(iterNum, round((path(ant,iterNum+1,1)-ybegin)/deltaY), round((path(ant,iterNum+1,2)-zbegin)/deltaZ))...
%             = (1 - rou) * pheromone(iterNum, round((path(ant,iterNum+1,1)-ybegin)/deltaY), round((path(ant,iterNum+1,2)-zbegin)/deltaZ)) + 1/cfit;
%         end
%     end
end
%% 绘制最佳路径以及适应值变化图
% 绘制路径
x = [start(1):deltaX:goal(1)-0.001,goal(1)];
[~,m] = size(x);
path_ = [];
for i = 1:m
    path_ = [path_;bestPath(:,i,1),bestPath(:,i,2)];
end
bar3 = plot3(x, path_(:,1), path_(:,2),'LineWidth',2,'MarkerSize',7,'Color','r');
filPathX = [start(1),MovingAverage(x(2:end-1),5),goal(1)];
filPathY = [start(2),MovingAverage(path_(2:end-1,1),5),goal(2)];
filPathZ = [start(3),MovingAverage(path_(2:end-1,2),5),goal(3)];
bar4 = plot3(filPathX, filPathY, filPathZ,'LineWidth',3,'color','g');
legend([bar1, bar2,bar3,bar4],["起始点","终止点","无人机航迹","MA平滑后航迹"],'Location','northwest');
% 最终path
xx = [start(1),x(2:end-1),goal(1)]; yy = [start(2),path_(2:end-1,1)',goal(2)]; zz = [start(3),path_(2:end-1,2)',goal(3)];
path = [xx',yy',zz'];
pathLength = 0;
for i=1:length(path(:,1))-1, pathLength = pathLength + getDist(path(i,1:3),path(i+1,1:3)); end
fprintf('计算时间：%f秒\n 路径长度:%f\n GS:%f°\n LS:%f°',toc,pathLength,calGs(path)/pi*180,calLs(path)/pi*180);
% 绘制适应值变化图
figure(2)
plot(bestFitness,'LineWidth',2,'Color','r'); hold on;
plot(everyIterFitness,'LineWidth',2,'Color','b')
legend('历史最佳个体适应度','每代最佳个体适应度')
title('适应度变化趋势'); xlabel('迭代次数'); ylabel('适应度值'); grid on;
%% 存储轨迹
csvwrite('F:\MasterDegree\毕业设计\实验数据\静态环境轨迹联合绘制/Ant_path.csv',[filPathX',filPathY', filPathZ']);
%% 函数
function [path, pheromone] = searchPath(popNumber, pheromone, start, goal, ycMax, zcMax,...
                                        deltaX, deltaY, deltaZ, obstacleMatrix, RobstacleMatrix,...
                                        cylinderMatrix, cylinderRMatrix, cylinderHMatrix,...
                                        coneMatrix, coneRMatrix, coneHMatrix,...
                                        ybegin, zbegin, gridYNumber, gridZNumber)
% 获取从起点到终点的路径函数
path = []; % 用于记录所有蚂蚁的路径
for ant = 1:popNumber % 对于每一只蚂蚁
    path(ant, 1, 1:2) = start(2:3); % 只记录y和z轴坐标，x轴每次加deltaX
    nowPoint = start(2:3);
    iterNum = 0;
    for x = start(1) + deltaX : deltaX : goal(1) - 0.001 % 减去一个小数避免x直接取到goal(1)
        iterNum = iterNum + 1;
        nextPoint = [];
        p = [];   
        for y = -ycMax * deltaY : deltaY : ycMax * deltaY
            for z = -zcMax * deltaZ : deltaZ : zcMax * deltaZ
                nextPoint = [nextPoint; nowPoint + [y, z]];
                if nextPoint(end,1) > ybegin+0.01 && nextPoint(end,1) < ybegin + gridYNumber*deltaY && ...
                   nextPoint(end,2) > zbegin+0.01 && nextPoint(end,2) < zbegin + gridZNumber*deltaZ  % 判断是否越界（信息素矩阵大小已经定了，避免超出）
                    hValue = calHeuristicValue(nowPoint, nextPoint(end,:), goal, x, deltaX, obstacleMatrix,...
                                               RobstacleMatrix, cylinderMatrix, cylinderRMatrix, cylinderHMatrix,...
                                               coneMatrix, coneRMatrix, coneHMatrix);
%                     pher = pheromone(iterNum, round((nextPoint(end,1) - ybegin)/deltaY), round((nextPoint(end,2) - zbegin)/deltaZ));
                    try
                        pher = pheromone(iterNum, round((nextPoint(end,1) - ybegin)/deltaY), round((nextPoint(end,2) - zbegin)/deltaZ));
                    catch
                        round((nextPoint(end,1) - ybegin)/deltaY)
                    end
                    p = [p, pher * hValue];
                else
                    p = [p,0]; %置零在轮盘赌中不可能被选中
                end
            end
        end
        % 轮盘赌选择下一坐标点
        p1 = p / sum(p); % 归一化
        pc = cumsum(p1);
        targetIndex = find(pc >= rand);
        targetNextPoint = nextPoint(targetIndex(1),:);
        path(ant, iterNum + 1, 1:2) = targetNextPoint;
        nowPoint = targetNextPoint;
    end
    path(ant, iterNum + 2, 1:2) = goal(2:3);
end
end

function h = calHeuristicValue(now, next, goal, x, deltaX, obstacleMatrix, RobstacleMatrix,...
                               cylinderMatrix, cylinderRMatrix, cylinderHMatrix,...
                               coneMatrix, coneRMatrix, coneHMatrix)
% 判断下一个坐标点是否碰撞，若碰撞则将启发值置为0，在后续的轮盘赌点位选择时将不可能被选中
nextXYZ = [x, next];
flag = checkCol(nextXYZ, obstacleMatrix, RobstacleMatrix,...
                cylinderMatrix, cylinderRMatrix, cylinderHMatrix,...
                coneMatrix, coneRMatrix, coneHMatrix);
% 计算启发值
d1 = getDist([x - deltaX, now], [x, next]);
d2 = getDist([x, next], goal);
D = 50 / (d1 + d2);
h = flag * D;
end

function f = calFit(path, deltaX, start, goal)
% 计算适应值函数
[n,m,~] = size(path);
x = [start(1) : deltaX : goal(1) - 0.001, goal(1)];
for i = 1:n
    f(i) = 0;
    for j = 1:m-1
        f(i) = f(i) + getDist([x(j), path(i,j,1), path(i,j,2)], [x(j+1), path(i,j+1,1), path(i,j+1,2)]);
    end
end
end

function flag = checkCol(pos, circleCenter,circleR, cylinderCenter,cylinderR, cylinderH,...
                         coneMatrix, coneRMatrix, coneHMatrix)
% 碰撞检测函数
[numberOfSphere, ~] = size(circleCenter);
[numberOfCylinder, ~] = size(cylinderCenter);
[numberOfCone,~] = size(coneMatrix);
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

function d = getDist(x,y)
d = sqrt(sum((x - y).^2));
end







































