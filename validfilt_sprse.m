%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function G = validfilt_sprse(A,B)
% f = validfilt_sprse(A,B);
%
% Convolve sparse A with each frame of B and return only "valid" part

am = size(A,1);
[bm,~,nflt] = size(B);
nn = am+bm-1;
npre = bm-1;
npost = bm-1;

% Do convolution
% G = zeros(nn,nflt);
% for i = 1:nflt
%     Y = A*B(:,:,i)'; 
%     for j = 1:bm
%         G(j:j+am-1,i) = G(j:j+am-1,i) + Y(:,bm-j+1);
%     end
% end
% G = G(npre+1:nn-npost,:);

G=zeros(am,nflt);
for i = 1:nflt
    for j=1:size(B,2)
        G(:,i)=G(:,i)+filter(flipud(squeeze(B(:,j,i))),1,A(:,j));
    end
end
G(1:bm-1,:)=[];

end