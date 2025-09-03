function [nll,del_nll,output] = pred_triad_resp(coords,beh,lamS,lamF,fit_cond,grad)

n = length(beh);
m = size(coords,2);

% make sure there are 3 attention conditions
if size(coords,3)==1
    coords = repmat(coords,[1 1 3 1]);
end

% group triad-only space
G = mean(coords(:,:,end,:),4);

% normalize coordinate space
proj_coords = coords ./ mean(vecnorm(coords,2,2),1);
proj_G = mean(proj_coords(:,:,end,:),4);

% fit separate condition coordinates?
if fit_cond
    C = mean(coords,4) - G;
    proj_C = mean(proj_coords,4) - proj_G;
else
    C = zeros(36,m,3,n);
    proj_C = zeros(36,m,3,n);
end

% individual subject ridge
if isinf(lamS)
    S = zeros(36,m,3,n);
    lossS = 0;
    del_S = 0;
else
    S = coords - G - C;
    proj_S = proj_coords - proj_G - proj_C;
    lossS = lamS.*sum(proj_S(:).^2,'all');
    del_S = 2 .* lamS .* proj_S;
end

temp_coords = G + C + S; % rebuild coordinate space
this_coords = reshape(permute(temp_coords,[1 3 2 4]),3*36,m,n);

% fusion ridge
if isinf(lamF)
    lossF = 0;
    del_F = 0;
else
    Sc = diff([temp_coords(end,:,:,:); temp_coords],[],1);
    lossF = lamF .* sum(Sc.^2,'all');
    del_F = 2 .* lamF .* (Sc - circshift(Sc,-1,1));
end

%% calcualte negative log-likelihood for these coords + data

% predict triad responses for each subject
ns = nan(1,n);
all_nll = nan(1,n);
all_p = nan(1,n);

for ii=1:n
    ns(ii) = length(beh(ii).triadRefA);
    beh(ii).sid = ii.*ones(1,ns(ii));
end

all_triadRef = [beh.triadRefA];
all_triadChoice = vertcat(beh.triadChoiceA);
all_triadChosenLoc = [beh.triadChosenLoc];
all_sid = [beh.sid];

y = all_triadChosenLoc==1; % response vector

full_nll = nan(size(this_coords));
nTrials = length(all_triadRef);

% get distance matrix and trial indices
nStim = size(this_coords,1);
Gs = pagemtimes(this_coords,pagetranspose(this_coords));
distMat = sum(Gs.*eye(nStim),1) + sum(Gs.*eye(nStim),2) - 2.*Gs;
distMat(distMat<=0) = 1e-20; % fix negative distances
distMat = sqrt(distMat);
iL = sub2ind(size(distMat),all_triadRef,all_triadChoice(:,1)',all_sid);
iR = sub2ind(size(distMat),all_triadRef,all_triadChoice(:,2)',all_sid);

% get choice probabilities based on distances of current coordinates
p = normcdf(sqrt(2).*(distMat(iR)-distMat(iL)));
p(p<1e-16) = 1e-16; p(p>1-1e-16) = 1-1e-16;

for ii=n:-1:1
    all_nll(ii) = -(log2(p(all_sid==ii))*(y(all_sid==ii))' + log2(1-p(all_sid==ii))*(1-y(all_sid==ii))');
    all_p(ii) = mean((1+(p(all_sid==ii)<.5))==beh(ii).triadChosenLoc);
end

% add regularlized loss
nll = sum(all_nll./ns).*mean(ns) + lossS + lossF;

%% calculate gradients, if requested
if grad
    % to get the gradient of the probability in each direction, we need to
    % calculate these separately for trials in which each stimulus was shown in
    % each of the three positions, and then recombine them into a single matrix
    coordsL = this_coords(all_triadChoice(:,1)' + (all_sid-1)*nStim*m + nStim.*((1:m)-1)');
    coordsR = this_coords(all_triadChoice(:,2)' + (all_sid-1)*nStim*m + nStim.*((1:m)-1)');
    coordsC = this_coords(all_triadRef + (all_sid-1)*nStim*m + nStim.*((1:m)-1)');

    % use sparse matrices for speed/memory savings
    stimIdx = [bsxfun(@plus,all_triadChoice(:,1),nStim.*((1:m)-1)) ...
        bsxfun(@plus,all_triadChoice(:,2),nStim.*((1:m)-1)) ...
        bsxfun(@plus,all_triadRef',nStim.*((1:m)-1))]';
    dL = (coordsC-coordsL)./distMat(iL);
    dR = (coordsR-coordsC)./distMat(iR);
    tempAll = sparse(stimIdx,repmat(1:nTrials,m*3,1),[dL; dR; -dL-dR]);

    % scale the delta-p (chain rule things)
    del_p = tempAll .* normpdf(sqrt(2).*(distMat(iR)-distMat(iL))).*sqrt(2);

    % compute the change in NLL along each dimension
    p1 = 1./p(y); p2 = 1./(1-p(~y));
    del_nll_p(:, y) = -p1./log(2); %-(((all_triadChosenLoc==1)./p)./log(2) + ... -((all_triadChosenLoc==2)./(1-p))./log(2));
    del_nll_p(:,~y) =  p2./log(2);
    sub_nll = del_p .* del_nll_p;

    for ii=1:n
        full_nll(:,:,ii) = reshape(sum(sub_nll(:,all_sid==ii),2),nStim,m);
    end

    del_nll = permute(reshape(full_nll,36,3,m,n),[1 3 2 4]);
    del_nll = del_nll .* shiftdim(mean(ns) ./ ns,-2);
    del_nll = del_nll + del_S + del_F;

    if ~fit_cond
        del_nll = sum(del_nll,3);
    end

    if isinf(lamS)
        del_nll = sum(del_nll,4);
    end
else
    del_nll = [];
end

output.nll = nll;
output.nll_grad = del_nll;
output.nll_sub = all_nll;
output.acc_sub = all_p;
output.lossS = lossS;
output.lossF = lossF;
