t   `clc
clear 
close all
%% send_part
% 参数设置
fs = 100e3; % 采样速率
sr = 5e3; % 码元速率
%%%%%%%%%%%%%%%%%%%%%%%%%创建发送对象%%%%%%%%%%%%%%%%%%%%%%%%%%%
txPluto = sdrtx('Pluto','RadioID','usb:0','CenterFrequency',800*1e6, ...
    'BasebandSampleRate',fs,'ChannelMapping',1,'Gain',0);
%%%%%%%%%%%%%%%%%%图像数据获取和编码%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J1=imread('xidian.png');
J=rgb2gray(J1);
imshow(J);
BinSer=dec2bin(J,8);%进行8位pcm编码
BinSer=BinSer';%转置方便后面处理数据
str=BinSer(:)' ;%将数据矩阵变换为一个列向量
leng=length(str);%列向量的长度
b=double(str);%用一个double类型矩阵来存储列向量str的数据
%这里注意char类型的直接用double转换是获得ASCII码的数值不是1和0的数值，但是str2num不能用于矩阵因此下面
%用一个循坏来处理
for i=1:leng
 b(i)=str2double(char(b(i)));
end    %循坏完成数据转换

bitData=b;
nBytes = length(str) * 1;
nCode = 1; %一个字节用8个bit编码

% 2. 进行交织
%%%%%%%%%%%%%%%%%信道编码（交织）%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bitData_scramble = matintrlv(bitData,10*nCode,nBytes/10);   %根据bigData来修改  10->12  
  %一个突发的错误变成随机错误，提升纠错能力
% 3. 映射为QPSK符号
%%%%%%%%%%%%%%%%%映射为QPSK符号，调制%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
symData = QPSKMap(bitData_scramble);
%  添加训练符号
symData = [ones(1, 100), symData];
% 4. 调制
modData = pskmod(symData, 4, pi/4);
figure(2)
scatter(real(modData),imag(modData));
% 5. 利用zc序列生成同步头
onePreData = CreatZC(500);
preData = repmat(onePreData, 1, 2);
preLength = length(preData);

% 6. 添加导频纠正相偏   无
totalData = modData;

% 7. 设置发射滤波器,也就是脉冲成型滤波器

irfn = 8;%；；滤波器截断的码元范围
ipoint = fs/sr;%每个码元范围内采样的点数 20
alfa = 0.5; % 由此可以得出带宽是10e3 * (1.5) = 15e3;
sendFilter = rcosdesign(alfa, irfn, ipoint, 'sqrt');
delay = irfn * ipoint / 2;
upData = upsample(totalData, ipoint);%进行上采样才能得到正确结果
toSend = conv(upData, sendFilter,'same');
%figure(2);
%plot(toSend);
%title('send waveform');
%grid on;

 %%%%%%%%%%%%%%%%pluto发送的数据要为列向量%%%%%%%%%%%%%%%%%%%%

toSend=[preData toSend];
toSend=toSend.';
%scatterplot(toSend);
%title('发送数据的散点图');
txPluto.transmitRepeat(toSend);%toSend
% while(true)
%txPluto(toSend);
%end

%% receive_part
%读取原始图片做对比时使用
J1=imread('xidian.png');
J=rgb2gray(J1);%真彩转灰度
x1=size(J,1);%读取行数
x2=size(J',1);%读取列数
totaldata=x1*x2*4+100;
BinSer=dec2bin(J,8);%十进制转二进制
BinSer=BinSer';%进行转置
str=BinSer(:)' ;%使其变成矩阵中每列合并成一列向量
leng=length(str);%读取其数组中的长度
b=double(str);

for i=1:leng
 b(i)=str2double(char(b(i)));%将其中字符串转换为数值
end
bitData =b;

rate1=[];

fs = 100e3;
rxPluto = sdrrx('Pluto','RadioID','usb:0','CenterFrequency',800e6, ...
    'BasebandSampleRate',fs,'ChannelMapping',1,'OutputDataType',...
    'double','Gain',20,'SamplesPerFrame',12e5);%对信号进行接收对象的创建


pre = CreatZC(500);
ipoint = 20;
irfn = 8;
alfa = 0.5; % 由此可以得出带宽是10e3 * (1.5) = 15e3;
recvFilter = rcosdesign(alfa, irfn, ipoint, 'sqrt');%（滚降系数，截断的符号范围，单个符号范围的采样点数，有sqrt时返回一个平方根升余弦滤波器）

freqComp = comm.CoarseFrequencyCompensator(...
    'Modulation','QPSK', ...
    'SampleRate',fs, ...
    'FrequencyResolution',1);%粗频偏纠正
symbolSync = comm.SymbolSynchronizer('SamplesPerSymbol', ipoint);%符号同步器，纠正码元时延，定时同步
fine = comm.CarrierSynchronizer( ...
    'SamplesPerSymbol',1,'Modulation','QPSK');%纠正载波频偏   载波同步器

i=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%循环接收 选择误码率较低%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while(true)
[data,datavalid,overflow] = rxPluto();

out = xcorr(data,pre);%计算互相关和自相关的函数
out = out(length(data):end);

figure(3)
plot(abs(out));

pos = find(abs(out) > 100);

[pos_max,loc] = max(abs(out(pos(1):pos(1)+20)));
pos1 = loc+pos(1)-1+500*2;

data = data(pos1:20*totaldata+pos1-1 );
%scatterplot(data);
%title('接收数据的散点图');
%data = agc(data);

[rxData1, estFreqOffset] = freqComp(data);
 estFreqOffset;   %估计频率偏移

 %scatterplot(rxData1);
 %title('粗频偏纠正后的散点图');

            % 匹配滤波
rxData2 = conv(rxData1, recvFilter, 'same');%求卷积0
% 符号同步 定时同步
rxData3 = symbolSync(rxData2);
if(length(rxData3) < totaldata)
    rxData3 = [rxData3', zeros(1, totaldata- length(rxData3))]';
else
    rxData3 = rxData3(1:totaldata);
end
%scatterplot(rxData3);
%title('符号同步后的散点图');

rxData4 = fine(rxData3);%相偏修正
%scatterplot(rxData4);
%title('相偏纠正后的散点图');

% 本来是取前100个训练符号来做纠正相偏，
pre100 = rxData3(1: 100);
%pre100=pre100(51:end);
pre100=pre100.';

a=repmat(-1+1j,1,100);
anout = angle(pre100./a);
anout = mean(anout);

 %scatterplot(a);

rxData4 = rxData3./exp(1j*anout);
rxData4=rxData4(101:end);

% scatterplot(rxData4);

% rxData4=rxData4.*exp(1j*pi/2);
demodData = pskdemod(rxData4, 4, pi/4);

% 解交织， 这里有一个符号映射bit的问题
bitDemap = QPSKDeMap(demodData);

nBytes = x1*x2*8 ;
nCode = 1; %一个字节用8个bit编码
bitData_descramble = matdeintrlv(bitDemap, 10* nCode, nBytes / 10);
% 比较误码率
%[~, rate] = biterr(demodData, symData');
[~, rate] = biterr(bitData, bitData_descramble);
rate;
rate1(i)=rate
i=i+1;
if(rate<0.015)  % 误码率
    break
end
end

str = num2str(bitData_descramble);
str1=strrep(str,' ','');
b=reshape(str1,8,x1*x2);
c=bin2dec(b');
d=reshape(c,x1,x2);
imshow(uint8(d));
%img_med = medfilt2(uint8(d), [2,2]);
%figure;imshow(img_med);title('2*2中值滤波');
scatterplot(rxData1);
title('粗频偏纠正后的散点图');
scatterplot(rxData3);
title('符号同步后的散点图');
scatterplot(rxData4);
title('相偏纠正后的散点图');
plot(rate1,'s-','linewidth',3);
title('误码率曲线');




