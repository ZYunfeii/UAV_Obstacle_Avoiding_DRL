function f = attraction(q,qgoal,dgoal,epsilon)
    r = distanceCost(q,qgoal);
    if r <= dgoal
        fx = epsilon * (qgoal(1) - q(1));
        fy = epsilon * (qgoal(2) - q(2));
    else
        fx = dgoal * epsilon * (qgoal(1) - q(1)) / r;
        fy = dgoal * epsilon * (qgoal(2) - q(2)) / r;
    end
    f = [fx,fy];
end

