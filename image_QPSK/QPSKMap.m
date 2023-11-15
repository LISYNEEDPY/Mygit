function symData = QPSKMap(bitData)
bitData = reshape(bitData, 2, length(bitData) / 2);
symData = bi2de(bitData');
symData = symData';
end