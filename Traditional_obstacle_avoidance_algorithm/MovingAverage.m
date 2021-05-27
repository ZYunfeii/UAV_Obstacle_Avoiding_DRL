function res = MovingAverage(input,N)
%% input为平滑前序列(列向量和行向量均可)；N为平滑点数（奇数）；res返回平滑后的序列(默认行向量)。
sz = max(size(input));
n = (N-1)/2;
res = [];
for i = 1:length(input)
    if i <= n
        res(i) = sum(input(1:2*i-1))/(2*i-1);
    elseif i < length(input)-n+1
        res(i) = sum(input(i-n:i+n))/(2*n+1);
    else
        temp = length(input)-i+1;
        res(i) = sum(input(end-(2*temp-1)+1:end))/(2*temp-1);
    end
end
end

