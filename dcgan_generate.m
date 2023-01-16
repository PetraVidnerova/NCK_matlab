function gen_imgs = dcgan_generate
  persistent gen;
  if isempty(gen)    
    gen = coder.loadDeepLearningNetwork('gen_dcgan_GPU.mat');
  end  
  r=100;c=100;
  %---
  %z = randn(100,1,1,r*c); 
  z = randn(3,4,100,r*c);
  z = dlarray(single(z),'SSCB');
  %---  
  gen_imgs = gen.predict(z);  
  gen_imgs = extractdata(gen_imgs);
  gen_imgs = reshape(gen_imgs,48,64,r*c);
end