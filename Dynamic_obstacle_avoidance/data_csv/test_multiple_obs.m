% 对多障碍环境的路径绘制
clc;clear;close all;
%% 读取数据
pathMatrix = csvread("./pathMatrix.csv");
obs1_trace = csvread("./obs0_trace.csv");
obs2_trace = csvread("./obs1_trace.csv");
obs3_trace = csvread("./obs2_trace.csv");
obs4_trace = csvread("./obs3_trace.csv");
obs_r_list = csvread("./obs_r_list.csv");
start = csvread("./start.csv");
goal = csvread("./goal.csv");
%% 绘制
scatter3(start(1),start(2),start(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')
text(start(1),start(2),start(3),'  起点');
text(goal(1),goal(2),goal(3),'  终点');
xlabel('x(km)'); ylabel('y(km)'); zlabel('z(km)');
% title('UAV动态航迹规划路径'); 
axis equal;
%% 主循环
[n,~] = size(pathMatrix);
for i = 1:n-1
    obsCenter1 = [obs1_trace(i,1),obs1_trace(i,2),obs1_trace(i,3)];
    obsCenter2 = [obs2_trace(i,1),obs2_trace(i,2),obs2_trace(i,3)];
    obsCenter3 = [obs3_trace(i,1),obs3_trace(i,2),obs3_trace(i,3)];
    obsCenter4 = [obs4_trace(i,1),obs4_trace(i,2),obs4_trace(i,3)];
    try delete(H1), catch, end
    try delete(H2), catch, end
    try delete(H3), catch, end
    try delete(H4), catch, end
    try delete(B2), catch, end
    H1 = drawSphere(obsCenter1, obs_r_list(1));
    H2 = drawSphere(obsCenter2, obs_r_list(2));
    H3 = drawSphere(obsCenter3, obs_r_list(3));
    H4 = drawSphere(obsCenter4, obs_r_list(4));
    B2 = scatter3(pathMatrix(i,1),pathMatrix(i,2),pathMatrix(i,3),80,'filled',"^",'MarkerFaceColor','g'...
                  ,'MarkerEdgeColor','k');
    if i >1
        b1 = plot3([obs1_trace(i-1,1),obs1_trace(i,1)],[obs1_trace(i-1,2),obs1_trace(i,2)]...
            ,[obs1_trace(i-1,3),obs1_trace(i,3)],'LineWidth',2,'color','k');
        b2 = plot3([obs2_trace(i-1,1),obs2_trace(i,1)],[obs2_trace(i-1,2),obs2_trace(i,2)]...
            ,[obs2_trace(i-1,3),obs2_trace(i,3)],'LineWidth',2,'color','k');
        b3 = plot3([obs3_trace(i-1,1),obs3_trace(i,1)],[obs3_trace(i-1,2),obs3_trace(i,2)]...
            ,[obs3_trace(i-1,3),obs3_trace(i,3)],'LineWidth',2,'color','k');
        b3 = plot3([obs4_trace(i-1,1),obs4_trace(i,1)],[obs4_trace(i-1,2),obs4_trace(i,2)]...
            ,[obs4_trace(i-1,3),obs4_trace(i,3)],'LineWidth',2,'color','k');
%           n1 = obs1_trace(i,:) - obs1_trace(i-1,:);
%           n2 = obs2_trace(i,:) - obs2_trace(i-1,:);
%           n3 = obs3_trace(i,:) - obs3_trace(i-1,:);
%           n4 = obs4_trace(i,:) - obs4_trace(i-1,:);
%           drawCircle3D(obs1_trace(i,:),n1,obs_r_list(1));
%           drawCircle3D(obs2_trace(i,:),n2,obs_r_list(2));
%           drawCircle3D(obs3_trace(i,:),n3,obs_r_list(3));
%           drawCircle3D(obs4_trace(i,:),n4,obs_r_list(4));
    end
    drawnow;
    b2 = plot3([pathMatrix(i,1),pathMatrix(i+1,1)],[pathMatrix(i,2),pathMatrix(i+1,2)],[pathMatrix(i,3),pathMatrix(i+1,3)],'LineWidth',2,'Color','r');
    if i == 2
        legend([b2,B2],["UAV规划航路","UAV"],'AutoUpdate','off','Location','best')
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

% 三维圆绘制函数
function h = drawCircle3D(pos,n,r)
theta=(0:2*pi/100:2*pi)'; %theta角从0到2*pi
a=cross(n,[1 0 0]); %n与i叉乘，求取a向量
if ~any(a) %如果a为零向量，将n与j叉乘
    a=cross(n,[0 1 0]);
end
b=cross(n,a); %求取b向量
a=a/norm(a); %单位化a向量
b=b/norm(b); %单位化b向量

c1=pos(1)*ones(size(theta,1),1);
c2=pos(2)*ones(size(theta,1),1);
c3=pos(3)*ones(size(theta,1),1);

x=c1+r*a(1)*cos(theta)+r*b(1)*sin(theta);%圆上各点的x坐标
y=c2+r*a(2)*cos(theta)+r*b(2)*sin(theta);%圆上各点的y坐标
z=c3+r*a(3)*cos(theta)+r*b(3)*sin(theta);%圆上各点的z坐标

h = plot3(x,y,z,"LineWidth",2,"Color","b");
end




