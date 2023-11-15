function [bitData] = QPSKDeMap(symData)
% 直接就是10进制转2进制就可以了
temp = de2bi(symData, 2);
temp = temp';
bitData = reshape(temp, 1, 2 * length(symData));
end