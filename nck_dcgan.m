clear all; close all; clc;

%load images
imgdir = "../ExtractedImages";
imgdir_mask = "../ExtractedImages" + "/*.jpg";
files = dir(imgdir_mask);
tables = {};
trainX = zeros(48,64,1,length(files));
for i = 1:length(files)
    name = imgdir + "/" + files(i).name;
    trainX(:,:,1,i) = imread(name);
end
size(trainX);
trainX = preprocess(trainX);

% The 64x48 Generator
gf_dim = 64;
gen_layers = [
    imageInputLayer([3,4,100],'Name','ginput','Normalization','none')
    convolution2dLayer([1,1],512,'Stride',[1,1],'Padding','same','Name','convfc')
    %---
    %imageInputLayer([100,1,1],'Name','ginput','Normalization','none')
    %fullyConnectedLayer(gf_dim*8*3*4, 'Name','fc')
    batchNormalizationLayer('Name','bn0')
    leakyReluLayer('Name','act0')
    %---
    %resize2dLayer('OutputSize',[224,224],'Name','resize224')
    %reshapeLayer('reshape')
    %projectAndReshapeLayer([3,4,512],6144,'reshape')
    %---
    transposedConv2dLayer([5,5],gf_dim*4,'Stride',[2,2],'Cropping','same','Name','tpconv1')
    batchNormalizationLayer('Name','bn1')
    leakyReluLayer('Name','act1')
    %---
    transposedConv2dLayer([5,5],gf_dim*2,'Stride',[2,2],'Cropping','same','Name','tpconv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer('Name','act2')
    %---
    transposedConv2dLayer([5,5],gf_dim*1,'Stride',[2,2],'Cropping','same','Name','tpconv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer('Name','act3')
    %---
    transposedConv2dLayer([5,5],1,'Stride',[2,2],'Cropping','same','Name','tpconv4')
    tanhLayer('Name','tanh')
    ];
generator = dlnetwork(layerGraph(gen_layers));
%analyzeNetwork(generator)

% The 64x48 Discriminator
gf_dim = 64;
disc_layers = [
    imageInputLayer([48,64,1],'Name','dinput','Normalization','none')
    convolution2dLayer([5,5],64,'Stride',[2,2],'Padding','same','Name','conv0')    
    leakyReluLayer('Name','dact0')    
    %---
    convolution2dLayer([5,5],128,'Stride',[2,2],'Padding','same','Name','conv1')    
    batchNormalizationLayer('Name','dbn1')
    leakyReluLayer('Name','dact1')
    %---
    convolution2dLayer([5,5],256,'Stride',[2,2],'Padding','same','Name','conv2')    
    batchNormalizationLayer('Name','dbn2')
    leakyReluLayer('Name','dact2')
    %---
    convolution2dLayer([5,5],512,'Stride',[2,2],'Padding','same','Name','conv3')    
    batchNormalizationLayer('Name','dbn3')
    leakyReluLayer('Name','dact3')
    %---    
    fullyConnectedLayer(1,'Name','dfc')
    sigmoidLayer('Name','sgm')
    ];
discriminator = dlnetwork(layerGraph(disc_layers));
%analyzeNetwork(discriminator)

%---
settings.latentDim = 100;
settings.batch_size = 32; settings.image_size = [48,64,1]; 
settings.lrD = 0.0002; settings.lrG = 0.0002; 
settings.beta1 = 0.5;settings.beta2 = 0.999;
avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];
%---
numIterations = floor(size(trainX,4)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
settings.maxepochs = 50;
while ~out
    tic; 
    trainXshuffle = trainX(:,:,:,randperm(size(trainX,4)));
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        %noise = gpdl(randn(settings.latentDim,1,1,settings.batch_size),'SSCB');
        noise = gpdl(randn(3,4,100,settings.batch_size),'SSCB');
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch=trainXshuffle(:,:,:,idx);
        XBatch=gpdl(single(XBatch),'SSCB');        

        [GradGen,GradDis,stateG] = dlfeval(@modelGradients,generator,discriminator,XBatch,noise);
        generator.State = stateG;

        % Update Discriminator network parameters
        [discriminator,avgG.Dis,avgGS.Dis] = ...           
            adamupdate(discriminator, GradDis, ...
            avgG.Dis, avgGS.Dis, global_iter, ...
            settings.lrD, settings.beta1, settings.beta2);

        % Update Generator network parameters
        [generator,avgG.Gen,avgGS.Gen] = ...
            adamupdate(generator, GradGen, ...
            avgG.Gen, avgGS.Gen, global_iter, ...
            settings.lrG, settings.beta1, settings.beta2);
        
        if i==1 || rem(i,20)==0
            progressplot(generator,discriminator,XBatch);
        end
        
    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end

save gen_dcgan_GPU.mat generator

function [GradGen,GradDis,stateG]=modelGradients(gen,disc,x,z)
[fake_images, stateG] = forward(gen,z);
d_output_real = forward(disc,x);
d_output_fake = forward(disc,fake_images);

% Flip labels.
flipFactor = 0.5;
numObservations = size(x,4);
idx = randperm(numObservations,floor(flipFactor * numObservations));
d_output_real(idx) = 1 - d_output_real(idx);

% Loss.
d_loss = -mean(log(d_output_real)+log(1-d_output_fake));
g_loss = -mean(log(d_output_fake));

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(g_loss,gen.Learnables,'RetainData',true);
GradDis = dlgradient(d_loss,disc.Learnables);
end

function dlx = gpdl(x,labels)
dlx = gpuArray(dlarray(x,labels));
end

function x = preprocess(x)
x = x / 127.5 - 1;
end

function x = gatext(x)
x = gather(extractdata(x));
x = (x+1)*127.5;
x = round(x);
x = min(max(x,0),255);
end

function progressplot(gen,disc,x)
r = 5; c = 5;
%z = gpdl(randn(100,1,1,r*c),'SSCB');
z = gpdl(randn(3,4,100,r*c),'SSCB');
gen_imgs = forward(gen,z);
gen_imgs = reshape(gen_imgs,48,64,[]);

fig = gcf;
if ~isempty(fig.Children)
    delete(fig.Children)
end

xx = gen_imgs;
I = imtile(gatext(xx));
imagesc(I)
title("Generated Images")
colormap gray

drawnow;
end