clear
close all
addpath('C:\Users\Wang\Desktop\libsvm-3.21\matlab');

%% training data and labels
training_data=[];

for pic_num=1:25
    if pic_num<10
        training_data=[training_data;['tr/tr0' num2str(pic_num) '.jpg']];
    else
        training_data=[training_data;['tr/tr' num2str(pic_num) '.jpg']];
    end
end
label_a=[1;1;1;1;1;0;0;0;0;0;1;1;1;0;0;0;1;1;1;0;1;1;0;1;0]; % 0: nonface, 1: face

features_a=zeros(size(training_data,1),3);
for pic_num=1:size(training_data,1)
    im1=double(imread(training_data(pic_num,:)));
    im1=rgb2ycbcr(im1);
    x=mean(mean(im1(:,:,:)));
    features_a(pic_num,:)=[x(1) x(2) x(3)];
end

%% scaling
[m1,N]=size(features_a);
mf=mean(features_a);
nrm=diag(1./std(features_a,1));
features_1=(features_a-ones(m1,1)*mf)*nrm;

%% SVM
model=svmtrain(label_a,features_1);
start_num=[3952,4014,4083,4159,4215,4269,4323,4384,4458,4522,4575,4676,4733,...
            4792,4844,4896,4948,5003,5065,4627];

%% test data and labels
for sign=1:1
    for pic_num=10:10
        test_data=['data/' num2str(sign) '/IMG_' num2str(start_num(sign)+pic_num-1) '.jpg'];
        for m=1:size(test_data,1)
            im1=double(imread(test_data(m,:)));
            im1=im1(1:2:479,1:2:639,:);
            im2=rgb2ycbcr(im1);
            features_b=reshape(im2,size(im2,1)*size(im2,2),3);
            label_b=floor(rand(size(im2,1)*size(im2,2),1)+0.5);
            % scaling
            [m2,N]=size(features_b);
            features_2=(features_b-ones(m2,1)*mf)*nrm;
            % test % predicted: the SVM output of the test data
            [predicted,accuracy,d_values]=svmpredict(label_b,features_2,model);
            % group the regions
            predicted=reshape(predicted,size(im2,1),size(im2,2));
            % skin picture
            figure
            imshow(predicted)
            colormap(gray(256))
            % bwlabel: group the regions
            A1=bwlabel(predicted);
            size1=zeros(max(max(A1)),1);
            for n=1:max(max(A1))
                [fr,fc]=find(A1==n);
                size1(n,1)=size(fr,1);
            end
            % find the 1st region and 2nd region 
            [M1,I1]=max(size1);
            size1(I1)=0;
            [M2,I2]=max(size1);
            %find the lower region
            Y_temp1=im2(:,:,1).*(A1==I1);
            Y_temp2=im2(:,:,1).*(A1==I2);
            [x,y]=find(Y_temp1~=0);
            mean1=mean(x);
            [x,y]=find(Y_temp2~=0);
            mean2=mean(x);
            if mean1>mean2
                Y1=Y_temp1;
            else
                Y1=Y_temp2;
            end
            [fr,fc]=find(Y1~=0);
            Y1=Y1(min(fr):max(fr),min(fc):max(fc));
            [height, width]=size(Y1);
            max_num=max(height,width);
            max_num=ceil(max_num/2)*2;
            Y2=zeros(max_num,max_num);
            Y2(((max_num/2)-ceil(height/2)+1):1:((max_num/2)-ceil(height/2)+height),...
                ((max_num/2)-ceil(width/2)+1):1:((max_num/2)-ceil(width/2)+width))=Y1;
            % saved hand picture
            figure
            imshow(Y2/255)
            % save data
            imwrite(Y2/255,['result/' num2str(sign) '/' num2str(sign) '_' num2str(pic_num) '.jpg'],'Quality',100)
            % hand picture
            R1=im1(:,:,1).*(A1==I2);
            G1=im1(:,:,2).*(A1==I2);
            B1=im1(:,:,3).*(A1==I2);
            figure
            imshow(cat(3,R1/255,G1/255,B1/255))
            colormap(gray(256))
        end
    end
end