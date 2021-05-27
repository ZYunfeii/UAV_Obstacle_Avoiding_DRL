% IFDS算法实现
%% 初始参数设置
V0 = 1; 
threshold = 0.2;
stepSize = 0.1;
%% demo1:静态单球形障碍
start = [0,1,5];
goal = [10,10,6];
obsCenter = [5,5,5.5];
obsR = 1;
row0 = 1;
% 绘制环境
scatter3(start(1),start(2),start(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')
text(start(1),start(2),start(3),'  起点');
text(goal(1),goal(2),goal(3),'  终点');
drawSphere(obsCenter,obsR);
xlabel('x(m)'); ylabel('y(m)'); zlabel('z(m)');
title('UAV动态航迹规划路径'); axis equal;
% 开始
pos = start;
path = pos;
while distanceCost(pos, goal) > threshold
    u = initField(pos, V0, goal);
    n = partialDerivativeSphere(obsCenter, pos, obsR);
    tempD = distanceCost(pos, obsCenter) - obsR; % 到障碍物表面距离
    row = row0 * exp(1-1/(distanceCost(pos, goal)*tempD));
    T = calculateT(obsCenter, pos, obsR);
    M = eye(3) - n * n' / T^(1/row) / (n'*n);
    ubar = (M * u)';
    nextPos = pos + ubar * stepSize;
    path = [path;nextPos];
    pos = nextPos;
end
path = [path;goal];
plot3(path(:,1), path(:,2), path(:,3),'LineWidth',2,'color','b'); hold on;

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
