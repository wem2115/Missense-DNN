function [x, i, j]=shuffle( y, dim)

if nargin < 1
	help shuffle;
	return;
end;
if nargin < 2
	if size(y,1) ~= 1
		dim = 1;
	else
		if size(y,2) ~= 1
			dim = 2;
		else
			dim = 3;
		end;
	end;
end;
	
r =size(y, dim);
a = rand(1,r);
[tmp i] = sort(a);
switch dim
	case 1
		x = y(i,:,:,:,:);
	case 2
		x = y(:,i,:,:,:);
	case 3
		x = y(:,:,i,:,:);
	case 4
		x = y(:,:,:,i,:);
	case 5
		x = y(:,:,:,:,i);
end;		
[tmp j] = sort(i); % unshuffle

return;
