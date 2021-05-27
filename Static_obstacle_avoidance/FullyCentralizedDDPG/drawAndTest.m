clc; clear; close all;
%% 读入Matrix
pathMatrix = csvread("./data_csv/pathMatrix.csv");
obstacleMatrix = csvread("./data_csv/obstacleMatrix.csv");
RobstacleMatrix = csvread("./data_csv/RobstacleMatrix.csv");
cylinderMatrix = csvread("./data_csv/cylinderMatrix.csv");
cylinderRMatrix = csvread("./data_csv/cylinderRMatrix.csv");
cylinderHMatrix = csvread("./data_csv/cylinderHMatrix.csv");
coneMatrix = csvread("./data_csv/coneMatrix.csv");
coneRMatrix = csvread("./data_csv/coneRMatrix.csv");
coneHMatrix = csvread("./data_csv/coneHMatrix.csv");
start = csvread("./data_csv/start.csv");
goal = csvread("./data_csv/goal.csv");
%% 绘图
figure(1)
[n,~] = size(obstacleMatrix);
for i = 1:n   %绘制静态球障碍物
    drawSphere(obstacleMatrix(i,:), RobstacleMatrix(i))
end

[n,~] = size(coneMatrix);
for i = 1:n
    drawCone(coneMatrix(i,:), coneRMatrix(i), coneHMatrix(i));
end

[n,~] = size(cylinderMatrix);
for i = 1:n   %绘制圆柱体障碍物
    drawCylinder(cylinderMatrix(i,:), cylinderRMatrix(i), cylinderHMatrix(i));
end

scatter3(start(1),start(2),start(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')
text(start(1),start(2),start(3),'  起点');
text(goal(1),goal(2),goal(3),'  终点');
xlabel('x(km)'); ylabel('y(km)'); zlabel('z(km)');
title('UAV航路规划路径');
%% 绘制path
%plot3(pathMatrix(:,1), pathMatrix(:,2), pathMatrix(:,3),'LineWidth',2,"Color",'r')

%动态path
[n,~] = size(pathMatrix);
view(30,60)
axis equal %放在绘图之后

for i = 1 : n - 1
    plot3([pathMatrix(i,1),pathMatrix(i+1,1)],[pathMatrix(i,2),pathMatrix(i+1,2)],...
        [pathMatrix(i,3),pathMatrix(i+1,3)],'LineWidth',2,"Color",'r');
    
%     [x,y,z] = sphere();
%     try
%         delete(dynamicSphere)
%     catch
%     end
%     dynamicSphere = surf(dynamicSphereR0*x+dynamicSpherePathMatrix(i,1),...
%                          dynamicSphereR0*y+dynamicSpherePathMatrix(i,2),...
%                          dynamicSphereR0*z+dynamicSpherePathMatrix(i,3));
    pause(0.05)
end
%% 计算指标
pathLength = 0;
for i=1:length(pathMatrix(:,1))-1, pathLength = pathLength + distanceCost(pathMatrix(i,1:3),pathMatrix(i+1,1:3)); end
fprintf('路径长度:%f\n GS:%f°\n LS:%f°',pathLength,calGs(pathMatrix)/pi*180,calLs(pathMatrix)/pi*180);
%% 所需函数
% 圆锥绘制函数
function drawCone(pos, r, h)
t = 0:1/20:1;
[x,y,z] = cylinder(r-r*t,40);
surf(x+pos(1),y+pos(2),h*z);hold on;
theta = linspace(0,2*pi,40);
X = r*cos(theta) + pos(1);
Y = r*sin(theta) + pos(2);
fill3(X,Y,zeros(size(X)),[0 0.5 1]);
end
% 球绘制函数
function drawSphere(pos, r)
[x,y,z] = sphere(60);
surfc(r*x+pos(1), r*y+pos(2), r*z+pos(3));
hold on;
end
% 圆柱体绘制函数
function drawCylinder(pos, r, h)
[x,y,z] = cylinder(r,40);
z(2,:) = h;
surfc(x + pos(1),y + pos(2),z,'FaceColor','interp');hold on;

theta = linspace(0,2*pi,40);
X = r * cos(theta) + pos(1);
Y = r * sin(theta) + pos(2);
Z = ones(size(X)) * h;
fill3(X,Y,Z,[0 0.5 1]); % 顶盖
fill3(X,Y,zeros(size(X)),[0 0.5 1]); % 底盖
end
% 计算欧式距离函数
function h=distanceCost(a,b)     
	h = sqrt(sum((a-b).^2, 2));
end



