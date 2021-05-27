function output = differential(q,other)
    output1 = (q(1) - other(1)) / distanceCost(q,other);
    output2 = (q(2) - other(2)) / distanceCost(q,other);
    output = [output1,output2];
end

