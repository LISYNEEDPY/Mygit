t   `clc
clear 
close all
%% send_part
% ��������
fs = 100e3; % ��������
sr = 5e3; % ��Ԫ����
%%%%%%%%%%%%%%%%%%%%%%%%%�������Ͷ���%%%%%%%%%%%%%%%%%%%%%%%%%%%
txPluto = sdrtx('Pluto','RadioID','usb:0','CenterFrequency',800*1e6, ...
    'BasebandSampleRate',fs,'ChannelMapping',1,'Gain',0);
%%%%%%%%%%%%%%%%%%ͼ�����ݻ�ȡ�ͱ���%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J1=imread('xidian.png');
J=rgb2gray(J1);
imshow(J);
BinSer=dec2bin(J,8);%����8λpcm����
BinSer=BinSer';%ת�÷�����洦������
str=BinSer(:)' ;%�����ݾ���任Ϊһ��������
leng=length(str);%�������ĳ���
b=double(str);%��һ��double���;������洢������str������
%����ע��char���͵�ֱ����doubleת���ǻ��ASCII�����ֵ����1��0����ֵ������str2num�������ھ����������
%��һ��ѭ��������
for i=1:leng
 b(i)=str2double(char(b(i)));
end    %ѭ���������ת��

bitData=b;
nBytes = length(str) * 1;
nCode = 1; %һ���ֽ���8��bit����

% 2. ���н�֯
%%%%%%%%%%%%%%%%%�ŵ����루��֯��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bitData_scramble = matintrlv(bitData,10*nCode,nBytes/10);   %����bigData���޸�  10->12  
  %һ��ͻ���Ĵ������������������������
% 3. ӳ��ΪQPSK����
%%%%%%%%%%%%%%%%%ӳ��ΪQPSK���ţ�����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
symData = QPSKMap(bitData_scramble);
%  ���ѵ������
symData = [ones(1, 100), symData];
% 4. ����
modData = pskmod(symData, 4, pi/4);
figure(2)
scatter(real(modData),imag(modData));
% 5. ����zc��������ͬ��ͷ
onePreData = CreatZC(500);
preData = repmat(onePreData, 1, 2);
preLength = length(preData);

% 6. ��ӵ�Ƶ������ƫ   ��
totalData = modData;

% 7. ���÷����˲���,Ҳ������������˲���

irfn = 8;%�����˲����ضϵ���Ԫ��Χ
ipoint = fs/sr;%ÿ����Ԫ��Χ�ڲ����ĵ��� 20
alfa = 0.5; % �ɴ˿��Եó�������10e3 * (1.5) = 15e3;
sendFilter = rcosdesign(alfa, irfn, ipoint, 'sqrt');
delay = irfn * ipoint / 2;
upData = upsample(totalData, ipoint);%�����ϲ������ܵõ���ȷ���
toSend = conv(upData, sendFilter,'same');
%figure(2);
%plot(toSend);
%title('send waveform');
%grid on;

 %%%%%%%%%%%%%%%%pluto���͵�����ҪΪ������%%%%%%%%%%%%%%%%%%%%

toSend=[preData toSend];
toSend=toSend.';
%scatterplot(toSend);
%title('�������ݵ�ɢ��ͼ');
txPluto.transmitRepeat(toSend);%toSend
% while(true)
%txPluto(toSend);
%end

%% receive_part
%��ȡԭʼͼƬ���Ա�ʱʹ��
J1=imread('xidian.png');
J=rgb2gray(J1);%���ת�Ҷ�
x1=size(J,1);%��ȡ����
x2=size(J',1);%��ȡ����
totaldata=x1*x2*4+100;
BinSer=dec2bin(J,8);%ʮ����ת������
BinSer=BinSer';%����ת��
str=BinSer(:)' ;%ʹ���ɾ�����ÿ�кϲ���һ������
leng=length(str);%��ȡ�������еĳ���
b=double(str);

for i=1:leng
 b(i)=str2double(char(b(i)));%�������ַ���ת��Ϊ��ֵ
end
bitData =b;

rate1=[];

fs = 100e3;
rxPluto = sdrrx('Pluto','RadioID','usb:0','CenterFrequency',800e6, ...
    'BasebandSampleRate',fs,'ChannelMapping',1,'OutputDataType',...
    'double','Gain',20,'SamplesPerFrame',12e5);%���źŽ��н��ն���Ĵ���


pre = CreatZC(500);
ipoint = 20;
irfn = 8;
alfa = 0.5; % �ɴ˿��Եó�������10e3 * (1.5) = 15e3;
recvFilter = rcosdesign(alfa, irfn, ipoint, 'sqrt');%������ϵ�����ضϵķ��ŷ�Χ���������ŷ�Χ�Ĳ�����������sqrtʱ����һ��ƽ�����������˲�����

freqComp = comm.CoarseFrequencyCompensator(...
    'Modulation','QPSK', ...
    'SampleRate',fs, ...
    'FrequencyResolution',1);%��Ƶƫ����
symbolSync = comm.SymbolSynchronizer('SamplesPerSymbol', ipoint);%����ͬ������������Ԫʱ�ӣ���ʱͬ��
fine = comm.CarrierSynchronizer( ...
    'SamplesPerSymbol',1,'Modulation','QPSK');%�����ز�Ƶƫ   �ز�ͬ����

i=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ѭ������ ѡ�������ʽϵ�%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while(true)
[data,datavalid,overflow] = rxPluto();

out = xcorr(data,pre);%���㻥��غ�����صĺ���
out = out(length(data):end);

figure(3)
plot(abs(out));

pos = find(abs(out) > 100);

[pos_max,loc] = max(abs(out(pos(1):pos(1)+20)));
pos1 = loc+pos(1)-1+500*2;

data = data(pos1:20*totaldata+pos1-1 );
%scatterplot(data);
%title('�������ݵ�ɢ��ͼ');
%data = agc(data);

[rxData1, estFreqOffset] = freqComp(data);
 estFreqOffset;   %����Ƶ��ƫ��

 %scatterplot(rxData1);
 %title('��Ƶƫ�������ɢ��ͼ');

            % ƥ���˲�
rxData2 = conv(rxData1, recvFilter, 'same');%����0
% ����ͬ�� ��ʱͬ��
rxData3 = symbolSync(rxData2);
if(length(rxData3) < totaldata)
    rxData3 = [rxData3', zeros(1, totaldata- length(rxData3))]';
else
    rxData3 = rxData3(1:totaldata);
end
%scatterplot(rxData3);
%title('����ͬ�����ɢ��ͼ');

rxData4 = fine(rxData3);%��ƫ����
%scatterplot(rxData4);
%title('��ƫ�������ɢ��ͼ');

% ������ȡǰ100��ѵ����������������ƫ��
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

% �⽻֯�� ������һ������ӳ��bit������
bitDemap = QPSKDeMap(demodData);

nBytes = x1*x2*8 ;
nCode = 1; %һ���ֽ���8��bit����
bitData_descramble = matdeintrlv(bitDemap, 10* nCode, nBytes / 10);
% �Ƚ�������
%[~, rate] = biterr(demodData, symData');
[~, rate] = biterr(bitData, bitData_descramble);
rate;
rate1(i)=rate
i=i+1;
if(rate<0.015)  % ������
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
%figure;imshow(img_med);title('2*2��ֵ�˲�');
scatterplot(rxData1);
title('��Ƶƫ�������ɢ��ͼ');
scatterplot(rxData3);
title('����ͬ�����ɢ��ͼ');
scatterplot(rxData4);
title('��ƫ�������ɢ��ͼ');
plot(rate1,'s-','linewidth',3);
title('����������');




