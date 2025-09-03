function [nll,del_nll] = pred_triad_distmat(beh,k,lam,input,grad)

n = length(beh);

if size(k,3)==1
    k = repmat(k,[1 1 3]);
    fit_nocond = true;
else
    fit_nocond = false;
end

loss_k = lam .* sum(k(:).^2); % ridge penalization

% predict triad responses for each subject
ns = nan(1,n);
all_nll = nan(1,n);

for ii=1:n
    ns(ii) = length(beh(ii).triadRefA);
end

% reconstruct distance matrix from lower triangle values
distMat0 = zeros(36,36,n,3);
distMat0(repmat(tril(true(36),-1),1,1,n,3)) = k;
distMat0 = distMat0 + pagetranspose(distMat0);
distMat = [distMat0(:,:,:,1)  zeros(36,36,n)    zeros(36,36,n);
            zeros(36,36,n)   distMat0(:,:,:,2)  zeros(36,36,n);
            zeros(36,36,n)    zeros(36,36,n)   distMat0(:,:,:,3)];

% get relevant input variables
if ~isempty(input)
    all_sid = input.all_sid;
    y = input.all_triadChosenLoc==1; % response vector
    iL = input.iL;
    iR = input.iR;
else
    all_sid = [beh.sid];
    all_triadRef = [beh.triadRefA];
    all_triadChoice = vertcat(beh.triadChoiceA);
    y = [beh.triadChosenLoc]==1;
    iL = sub2ind(size(distMat),all_triadRef,all_triadChoice(:,1)',all_sid);
    iR = sub2ind(size(distMat),all_triadRef,all_triadChoice(:,2)',all_sid);
end

% get choice probabilities based on distances of current coordinates
p = 0.5 .* erfc(distMat(iL)-distMat(iR));
p(p<1e-16) = 1e-16; p(p>1-1e-16) = 1-1e-16;

for ii=n:-1:1
    all_nll(ii) = -(log2(p(all_sid==ii))*(y(all_sid==ii))' + log2(1-p(all_sid==ii))*(1-y(all_sid==ii))');
end

nll = sum(all_nll./ns).*mean(ns) + loss_k;

%% calculate gradients, if requested
if grad
    % get change in choice probabilities by RDM index
    % nTrials = length(all_triadRef);
    % idxMat = sparse(iR,1:nTrials,1,numel(distMat),nTrials) - ...
    %     sparse(iL,1:nTrials,1,numel(distMat),nTrials);
    idxMat = input.idxMat;
    del_p = (sqrt(2).*normpdf(sqrt(2).*(distMat(iR)-distMat(iL)))) .* idxMat;

    % compute the change in NLL along each dimension
    p1 = 1./p(y); p2 = 1./(1-p(~y));
    del_nll_p(:, y) = -p1./log(2); %-(((all_triadChosenLoc==1)./p)./log(2) + ... -((all_triadChosenLoc==2)./(1-p))./log(2));
    del_nll_p(:,~y) =  p2./log(2);

    sub_nll0 = reshape(del_p * del_nll_p',size(distMat));
    sub_nll(:,:,:,1) = sub_nll0(1:36,1:36,:);
    sub_nll(:,:,:,2) = sub_nll0(37:72,37:72,:);
    sub_nll(:,:,:,3) = sub_nll0(73:end,73:end,:);
    sub_nll = sub_nll + pagetranspose(sub_nll);
    sub_nll = reshape(sub_nll,36*36,n,3);

    del_nll = sub_nll(tril(true(36),-1),:,:) .* mean(ns) ./ ns;

    % add ridge loss
    del_k = 2 .* lam .* k;
    del_nll = del_nll + del_k;

    if fit_nocond
        del_nll = sum(del_nll,3);
    end
else
    del_nll = [];
end
