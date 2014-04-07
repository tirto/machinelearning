    %% Load data

load('a5a_data.mat');
[n, d] = size(X)

%% Add column of 1's for b
X = [ X ones(n,1)];

%% SGD

C = 100; %slack constant
eta = 0.0001; %learning rate
del = 0.001; %convergence criteria

W = zeros(d+1, 1); %weight vector
%b = 0; %intercept term

del_W = zeros(d+1, 1); %gradient w.r.t. W
%del_b = 0; %gradient w.r.t. b

% Shuffle X and Y
perm = [randperm(n)];
Xs = X(perm,:);
Ys = Y(perm,:);

cost = calculateCost(Xs(:,1:d), Ys, W(1:d), W(d + 1), C);
sgdcosts = 5:1;
sgdtime = 5:1;
delta = 1;
total = tic;
for k = 1: n 
    itertime = tic;
    ind = (Ys(k,1)*(Xs(k,:)*W) < 1);
    del_W = W - C*Ys(k,1)*Xs(k,:)'*ind;
    %del_b = -C*Y(k,1)*ind;
    
    W = W - eta*del_W;
    %b = b - eta*del_b;
    
    newCost = calculateCost(Xs(:,1:d), Ys, W(1:d), W(d + 1), C);
    percentCostChange = abs(cost - newCost)*100/cost;
    delta = (delta + percentCostChange)/2;
    
    sgdtime(k)= toc(itertime);
    % Convergence
    cost = newCost;
    sgdcosts(k) = newCost;
    
    if(delta < del)
        break
    end
    
    
        
    
    %fprintf('%d : Cost: %f delta: %f\n', k,cost,delta);
    
end
fprintf('SGD : Cost at convergence = %f, no of iterations = %d',cost,k);
toc(total)


%% Grad descent

C = 100;
eta = 0.0000003;
del = 0.25;

W = zeros(d+1, 1); %weight vector
%b = 0; %intercept term

cost = calculateCost(X(:,1:d), Y, W(1:d), W(d + 1), C);
percentCostChange = 100.0;
gdcosts = zeros(5,1);
gdtime = 5:1;
iter = 0;
total = tic;
while(percentCostChange > del)
    itertime = tic;
    del_W = zeros(d + 1, 1); %gradient w.r.t. W
    %del_b = 0; %gradient w.r.t. b
    
    for k = 1: n 
        ind = (Y(k,1)*(X(k,:)*W) < 1);
        del_W = del_W - Y(k,1)*X(k,:)'*ind;
        %del_b = del_b - Y(k,1)*ind;
    end
    
    W = W - eta*(W + C*del_W);
    %b = b - eta*C*del_b;
    
    newCost = calculateCost(X(:,1:d), Y, W(1:d), W(d + 1), C);
    percentCostChange = abs(cost - newCost)*100/cost;
        
    cost = newCost;    
    iter = iter+1;
    gdtime(iter)= toc(itertime);
    gdcosts(iter) =  cost;
    fprintf('%d : Cost: %f percentagechange: %f\n', iter,newCost,percentCostChange);
end
fprintf('GD : Cost at convergence = %f, no of iterations = %d',cost,iter);
toc(total)
%% Mini batch Grad descent

C = 100;
eta = 0.000001;
del = 0.01;

% Shuffle X and Y
perm = [randperm(n)];
Xs = X(perm,:);
Ys = Y(perm,:);

W = zeros(d + 1, 1); %weight vector
%b = 0; %intercept term

cost = calculateCost(Xs(:,1:d), Ys, W(1:d), W(d + 1), C);
percentCostChange = 100.0;

batch_size = 10;
mbgdcosts = zeros(5,1);
mbgdtime = 5:1;

iter = 0;

delta = 0;
total = tic;
for z = 1:batch_size:n
    itertime = tic;
    del_W = zeros(d+1, 1); %gradient w.r.t. W
    %del_b = 0; %gradient w.r.t. b
    
    for k = z:z+batch_size 
        if k > n 
            break
        end
        ind = (Ys(k,1)*(Xs(k,:)*W) < 1);
        del_W = del_W - Ys(k,1)*Xs(k,:)'*ind;
        %del_b = del_b - Y(k,1)*ind;
    end
    
    W = W - eta*C*del_W;
    %b = b - eta*C*del_b;
    
    newCost = calculateCost(Xs(:,1:d), Ys, W(1:d), W(d + 1), C);
    percentCostChange = abs(cost - newCost)*100/cost;
    delta = (delta + percentCostChange)/2;
   
    
    cost = newCost;    
    iter = iter+1;
    mbgdtime(iter) = toc(itertime);
    % Convergence
    if(delta < del)
        break
    end 
    %fprintf('%d : Cost: %f percentagechange: %f\n', iter,cost,percentCostChange);
    mbgdcosts(iter) =  cost;
end
fprintf('MBGD : Cost at convergence = %f, no of iterations = %d',cost,iter);
toc(total)


%% Plot 
figure(1);
plot(sgdcosts, 'g');
hold on;
plot(gdcosts, 'b');
hold on;
plot(mbgdcosts,'r');
grid on;
axis auto;
xlabel('# of iterations');
ylabel('Cost')
title('Cost vs k');
legend('SGD', 'BGD', 'MBGD');

%% Plot 

for i=2:size(sgdtime, 2)
    sgdtime(i) = sgdtime(i) + sgdtime(i-1);
end

for i=2:size(gdtime, 2)
    gdtime(i) = gdtime(i) + gdtime(i-1);
end

for i=2:size(mbgdtime, 2)
    mbgdtime(i) = mbgdtime(i) + mbgdtime(i-1);
end
%%
figure(2);
plot(sgdtime, 'g');
hold on;
plot(gdtime, 'b');
hold on;
plot(mbgdtime,'r');
grid on;
axis auto;
xlabel('# of iterations');
ylabel('Time')
title('Time vs k');
legend('SGD', 'BGD', 'MBGD');
%%
% SGD : Regularization vs Error
eta = 0.0001; %learning rate

del = 0.001; %convergence criteria


c = [1,10,50,100,200,300,400,500];
[a,b] = size(c);

% train
Xtr = X(1:6000,:);
Ytr = Y(1:6000,:);

for i = 1:b
    C = c(i);
    

    W = zeros(d+1, 1); %weight vector
    %b = 0; %intercept term
    del_W = zeros(d+1, 1); %gradient w.r.t. W
    %del_b = 0; %gradient w.r.t. b
    k = 1; %sample num
    cost = calculateCost(Xtr(:,1:d), Ytr, W(1:d), W(d + 1), C);
    percentCostChange = 100.0;
    iter = 0;
    ntr = 6000;
    delta = 0;
    %while(percentCostChange > del)
    for k = 1: ntr 
        ind = (Ytr(k,1)*(Xtr(k,:)*W+b) < 1);
        del_W = W - C*Ytr(k,1)*Xtr(k,:)'*ind;
        %del_b = -C*Ytr(k,1)*ind;
        W = W - eta*del_W;
        %b = b - eta*del_b;
        %end
        newCost = calculateCost(Xtr(:,1:d), Ytr, W(1:d), W(d + 1), C);
        percentCostChange = abs(cost - newCost)*100/cost;
        cost = newCost;    
        %k = mod(k,ntr) + 1;
        iter = iter+1;
        
        delta = (delta + percentCostChange)/2;
        % Convergence

        if(delta < del)
            break
        end
    end
    
    
    %errortr(i) = sum(abs(Ytr - ((Xtr*W > 0)*2 - 1))/2)/ntr;
    %fprintf('C: %d : Training Error: %f \n', C,errortr(i));
    
    % Test Error
    Xte = X(6001:6414,:);
    Yte = Y(6001:6414,:);
    nte = 414;
    errorte(i) = sum(abs(Yte - ((Xte*W+b> 0)*2 - 1))/2)/nte;

    fprintf('C: %d : Validation Error: %f \n', C,errorte(i));
end
%%
figure(3);
plot(c,errorte);
%legend('Training','Test')
grid on;
axis auto;
xlabel('C');
ylabel('Error')
title('SGD : Error vs C');


