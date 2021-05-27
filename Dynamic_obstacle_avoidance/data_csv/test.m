clc;clear;close all;
pathMatrix = csvread("./pathMatrix.csv");
obsMatrix = csvread("./obs_trace.csv");
start = csvread("./start.csv");
goal = csvread("./goal.csv");
obsR = csvread("./obs_r.csv");           % 动态障碍物的半径
scatter3(start(1),start(2),start(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')
text(start(1),start(2),start(3),'  起点');
text(goal(1),goal(2),goal(3),'  终点');
xlabel('x(m)'); ylabel('y(m)'); zlabel('z(m)');
title('UAV动态航迹规划路径'); 
axis equal;
[n,~] = size(pathMatrix);
for i = 1:n-1
    obsCenter = [obsMatrix(i,1),obsMatrix(i,2),obsMatrix(i,3)];
    try delete(B1), catch, end
    try delete(B2), catch, end
    B1 = drawSphere(obsCenter, obsR);
    B2 = scatter3(pathMatrix(i,1),pathMatrix(i,2),pathMatrix(i,3),80,'filled',"^",'MarkerFaceColor','g'...
                  ,'MarkerEdgeColor','k');
    if i >1
        b1 = plot3([obsMatrix(i-1,1),obsMatrix(i,1)],[obsMatrix(i-1,2),obsMatrix(i,2)]...
              ,[obsMatrix(i-1,3),obsMatrix(i,3)],'LineWidth',2,'color','b');
    end
    drawnow;
    b2 = plot3([pathMatrix(i,1),pathMatrix(i+1,1)],[pathMatrix(i,2),pathMatrix(i+1,2)],[pathMatrix(i,3),pathMatrix(i+1,3)],'LineWidth',2,'Color','r');
    if i == 2
        legend([b1,b2,B2],["障碍物移动轨迹","UAV规划航路","UAV"],'AutoUpdate','off','Location','best')
    end
end
%% 计算GS,LS,L
pathLength = 0;
for i=1:length(pathMatrix(:,1))-1, pathLength = pathLength + distanceCost(pathMatrix(i,1:3),pathMatrix(i+1,1:3)); end
fprintf("航路长度为:%f\n GS:%f °\n LS:%f °",pathLength, calGs(pathMatrix)/pi*180, calLs(pathMatrix)/pi*180);
%% 函数
% 球绘制函数
function bar = drawSphere(pos, r)
[x,y,z] = sphere(60);
bar = surfc(r*x+pos(1), r*y+pos(2), r*z+pos(3));
hold on;
end
% 欧式距离求解函数
function h=distanceCost(a,b)
h = sqrt(sum((a-b).^2, 2));
end