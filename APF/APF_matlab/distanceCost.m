function h=distanceCost(a,b)         %% distanceCost.m
	h = sqrt(sum((a-b).^2, 2));
end
