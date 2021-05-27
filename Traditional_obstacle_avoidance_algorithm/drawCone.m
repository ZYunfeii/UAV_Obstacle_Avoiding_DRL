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

