%Code for creating and training custom network.
format long
i1 = xlsread('Data_99', 'B2:O295');
input = i1';                                                        % Input in normalized form for the network.
target1 = xlsread('Data_99','P2:P295');                %target are the theta1 values between [0 1] corresponding to stress and temperature
target = target1'; 
trainingFcn = 'trainlm';                                   %Defining training algorithm customized Levenberg-Marqurdt                                 
net1 = fitnet([9 6]);                                    %Creating feedforwardnetwork
net1.initFcn = 'initlay';                                % Initializing weights and biases
net1.layers{1}.initFcn = 'initnw';                       
net1.layers{2}.initFcn = 'initnw';
net1 = init(net1);
net1.layers{1}.transferFcn = 'tansig';                   % layer 1 transfer function tansig
net1.layers{2}.transferFcn = 'tansig';                  % layer 2 transfer function purelin
net1.layers{3}.transferFcn = 'purelin';
net1.divideParam.trainRatio = 75/100;                   % Distributing data fot training
net1.divideParam.valRatio = 0;
net1.divideParam.testRatio = 25/100;
net1.trainParam.goal = 0.000001;                            % Stopping criteria for training.
net1.trainParam.epochs= 1000;
[net1,tr] = train(net1,input,target);                       %training the network
[trainInd,valInd,testInd] = dividerand(295,0.75,0,0.25);                             
w1 = net1.iw{1};                                     % retrieves weight matrix from hidden layer number 1
w2 = net1.lw{2};                                     % retrieves weight matrix from hidden layer number 2
w3 = net1.lw{3,2};
b1 = net1.b{1};                                      % retrieves bias from hidden layer number 1
b2 = net1.b{2};                                      % retieves bias from hidden layer number 2
b3 = net1.b{3};                                      % retireves bias from output layer
output = net1(input);                                % output from the network for the 12 inputs
e = gsubtract(target,output); 
trOut = output(tr.trainInd);
tsOut = output(tr.testInd);
valOut = output(tr.valInd);
trtar = target(tr.trainInd);
tstar = target(tr.testInd);
valtar = target(tr.valInd);                           % error between target and output from network
performance = perform(net1,target,output);               % performance of the network w.r.t error
%view(net1)                                              % network architecture
%plotregression(target,output);                           % regression plot between target and output from the network
%p = net1(predict)
%plot(target,output,'o');grid on
