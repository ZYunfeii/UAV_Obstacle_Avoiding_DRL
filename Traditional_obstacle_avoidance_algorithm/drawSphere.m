% 球绘制函数
function drawSphere(pos, r)
[x,y,z] = sphere(60);
surfc(r*x+pos(1), r*y+pos(2), r*z+pos(3));
hold on;
end

