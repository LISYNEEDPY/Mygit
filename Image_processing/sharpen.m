function out=sharpen(filename)
I=filename;
I1=im2double(I)
%h1=fspecial('laplacian',1);%������˹����
%h1=fspecial('gaussian',[9 9])%��˹��ͨ�˲�����
h1=fspecial('unsharp');%�Աȶ���ǿ�˲���
%h1=fspecial('prewitt');%ˮƽ��Եǿ���˲���
h2=transpose(h1);
bw1=imfilter(I1,h1,'replicate');
bw2=imfilter(I1,h2,'replicate')
I_s1=I1-bw1;
%I_s2=I1-bw2;
figure
subplot(121)
imshow(I1)
subplot(122)
imshow(bw1);
%subplot(133)
%imshow(I_s1)
%subplot(234)
%imshow(I1)
%subplot(235)
%imshow(bw2);
%subplot(236)
%imshow(I_s2)
%figure
%imshow(bw1+bw2)

end